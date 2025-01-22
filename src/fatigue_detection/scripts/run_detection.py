import cv2
import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
import dlib
from imutils import face_utils
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Détection de fatigue en temps réel')
    parser.add_argument('--model_path', type=str, default='models/checkpoints/best_model.keras',
                      help='Chemin vers le modèle entraîné')
    parser.add_argument('--shape_predictor', type=str, 
                      default='shape_predictor_68_face_landmarks.dat',
                      help='Chemin vers le fichier de landmarks')
    parser.add_argument('--threshold', type=float, default=0.25,
                      help='Seuil du ratio des yeux')
    parser.add_argument('--camera', type=int, default=0,
                      help='Index de la caméra')
    return parser.parse_args()

def init_detectors(shape_predictor_path):
    """Initialise les détecteurs dlib"""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)
    return detector, predictor

def eye_aspect_ratio(eye):
    """Calcule le ratio d'aspect de l'œil"""
    # Distances verticales
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Distance horizontale
    C = np.linalg.norm(eye[0] - eye[3])
    # Calcul du ratio
    ear = (A + B) / (2.0 * C)
    return ear

def detect_eyes(image, detector, predictor):
    """Détecte les yeux et calcule leurs ratios"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    if len(rects) == 0:
        return None, None, None
    
    # Prendre le premier visage détecté
    rect = rects[0]
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    
    # Extraire les coordonnées des yeux
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    
    # Calculer les ratios
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    
    # Moyenne des deux yeux
    ear = (leftEAR + rightEAR) / 2.0
    
    # Extraire les régions des yeux
    left_eye_region = extract_eye_region(image, leftEye)
    right_eye_region = extract_eye_region(image, rightEye)
    
    return (left_eye_region, right_eye_region), ear, shape

def extract_eye_region(image, eye_points, padding=5):
    """Extrait la région de l'œil avec padding"""
    x1 = min(eye_points[:, 0]) - padding
    y1 = min(eye_points[:, 1]) - padding
    x2 = max(eye_points[:, 0]) + padding
    y2 = max(eye_points[:, 1]) + padding
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    
    return image[int(y1):int(y2), int(x1):int(x2)]

def preprocess_eyes(eye_regions, target_size=(224, 224)):
    """Prétraite les régions des yeux"""
    if not eye_regions or len(eye_regions) != 2:
        return None
    
    left_eye, right_eye = eye_regions
    
    # Redimensionner chaque œil
    left_eye = cv2.resize(left_eye, (target_size[0]//2, target_size[1]))
    right_eye = cv2.resize(right_eye, (target_size[0]//2, target_size[1]))
    
    # Combiner les yeux
    eyes_combined = np.hstack([left_eye, right_eye])
    eyes_combined = eyes_combined.astype(np.float32) / 255.0
    eyes_combined = np.expand_dims(eyes_combined, axis=0)
    
    return eyes_combined

def draw_results(image, shape, ear, prediction, fps=None, threshold=0.25):
    """Dessine les résultats sur l'image"""
    # Dessiner les points de repère
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    height, width = image.shape[:2]
    
    # Afficher le EAR
    cv2.putText(image, f"EAR: {ear:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Déterminer l'état des yeux basé sur EAR et prédiction
    is_closed = ear < threshold
    if prediction is not None:
        is_closed = is_closed # or prediction[0] > prediction[1]
    
    # Affichage de l'état
    state = "1" if is_closed else "0"
    color = (0, 0, 255) if is_closed else (0, 255, 0)
    
    cv2.putText(image, f"Fatigue: {state}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    if prediction is not None:
        conf = max(prediction[0], prediction[1])
        cv2.putText(image, f"Conf: {conf:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Alerte si yeux fermés
    if is_closed:
        cv2.putText(image, "FATIGUE DETECTEE!", 
                    (width//4, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if fps is not None:
        cv2.putText(image, f"FPS: {fps:.1f}", 
                    (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
    
    return image

def run_detection():
    args = parse_args()
    
    # Initialisation des détecteurs
    print("Initialisation des détecteurs...")
    detector, predictor = init_detectors(args.shape_predictor)
    
    # Chargement du modèle
    print(f"Chargement du modèle depuis {args.model_path}...")
    model_path = Path(args.model_path)
    if not model_path.exists():
        model_path = model_path.with_suffix('.keras')
    model = tf.keras.models.load_model(model_path)
    
    # Initialisation de la webcam
    print(f"Initialisation de la caméra {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la caméra {args.camera}")
    
    # Configuration de la caméra
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\nDétection en temps réel démarrée")
    print("Contrôles:")
    print("  - 'q': Quitter")
    print("  - 'r': Réinitialiser la détection")
    print(f"  - Seuil EAR: {args.threshold}")
    
    # Variables pour le calcul des FPS
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # Buffers pour lisser les prédictions
    ear_buffer = []
    prediction_buffer = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur de lecture de la caméra")
                break
            
            frame_count += 1
            
            # Calcul des FPS
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - start_time)
            
            # Détection des yeux et calcul du EAR
            eye_regions, ear, shape = detect_eyes(frame, detector, predictor)
            
            if eye_regions is not None and shape is not None:
                # Lissage du EAR
                ear_buffer.append(ear)
                if len(ear_buffer) > 3:
                    ear_buffer.pop(0)
                avg_ear = np.mean(ear_buffer)
                
                # Prétraitement et prédiction
                processed_eyes = preprocess_eyes(eye_regions)
                if processed_eyes is not None:
                    prediction = model.predict(processed_eyes, verbose=0)[0]
                    
                    # Lissage des prédictions
                    prediction_buffer.append(prediction)
                    if len(prediction_buffer) > 3:
                        prediction_buffer.pop(0)
                    avg_prediction = np.mean(prediction_buffer, axis=0)
                    
                    # Dessin des résultats
                    frame = draw_results(frame, shape, avg_ear, avg_prediction, fps, args.threshold)
            
            # Affichage
            cv2.imshow('Detection de fatigue', frame)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                ear_buffer.clear()
                prediction_buffer.clear()
                print("Détection réinitialisée")
            
    except Exception as e:
        print(f"Erreur: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nDétection arrêtée")

if __name__ == '__main__':
    run_detection()