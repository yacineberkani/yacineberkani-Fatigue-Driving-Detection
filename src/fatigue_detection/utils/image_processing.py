import cv2
import numpy as np

def extract_eye_region(frame, landmarks, eye_type='left'):
    """Extrait la région de l'œil à partir des points de repère"""
    if eye_type == 'left':
        eye_points = landmarks[0:2]
    else:
        eye_points = landmarks[2:4]
        
    x1 = min(p[0] for p in eye_points)
    y1 = min(p[1] for p in eye_points)
    x2 = max(p[0] for p in eye_points)
    y2 = max(p[1] for p in eye_points)
    
    # Ajoute une marge autour de l'œil
    margin = int((x2 - x1) * 0.5)
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(frame.shape[1], x2 + margin)
    y2 = min(frame.shape[0], y2 + margin)
    
    return frame[y1:y2, x1:x2]

def compute_head_pose(landmarks):
    """Calcule la pose de la tête à partir des points de repère"""
    # Points 3D du modèle
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nez
        (-225.0, 170.0, -135.0),  # Œil gauche
        (225.0, 170.0, -135.0),   # Œil droit
        (0.0, -330.0, -65.0)      # Bouche
    ])
    
    # Conversion des points 2D
    image_points = np.array([
        landmarks[4],     # Nez
        landmarks[0],     # Œil gauche
        landmarks[2],     # Œil droit
        landmarks[5]      # Bouche
    ], dtype=np.float32)
    
    # Calcul de la pose avec solvePnP
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    return rotation_vec, translation_vec 