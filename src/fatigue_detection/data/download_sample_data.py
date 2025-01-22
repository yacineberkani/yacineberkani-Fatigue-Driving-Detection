import gdown
import zipfile
from pathlib import Path
import shutil
import os
import cv2
import numpy as np
import sys
from pathlib import Path

# Ajout du chemin racine au PYTHONPATH
root_dir = str(Path(__file__).parent.parent.parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.fatigue_detection.data.prepare_emsr_data import split_dataset

def create_test_image(state):
    """Crée une image de test selon l'état (open/closed)"""
    image = np.ones((224, 224, 3), dtype=np.uint8) * 128  # Fond gris
    
    if state == 'open':
        # Simuler un œil ouvert avec un cercle blanc
        cv2.circle(image, (112, 112), 50, (255, 255, 255), -1)
    else:
        # Simuler un œil fermé avec une ligne horizontale
        cv2.line(image, (62, 112), (162, 112), (255, 255, 255), 10)
    
    return image

def download_sample_data():
    """Télécharge et extrait les données d'exemple"""
    # Création des dossiers
    data_dir = Path('data')
    custom_dir = data_dir / 'custom'
    
    for category in ['eyes', 'mouth']:
        for state in ['open', 'closed']:
            (custom_dir / category / state).mkdir(parents=True, exist_ok=True)
    
    # URLs des fichiers de données (à remplacer par vos propres URLs)
    sample_data_urls = {
        'eyes_open': 'https://drive.google.com/uc?id=YOUR_EYES_OPEN_ID',
        'eyes_closed': 'https://drive.google.com/uc?id=YOUR_EYES_CLOSED_ID',
        'mouth_open': 'https://drive.google.com/uc?id=YOUR_MOUTH_OPEN_ID',
        'mouth_closed': 'https://drive.google.com/uc?id=YOUR_MOUTH_CLOSED_ID'
    }
    
    # Téléchargement et extraction
    for data_type, url in sample_data_urls.items():
        category, state = data_type.split('_')
        output_dir = custom_dir / category / state
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"\nTéléchargement de {data_type}...")
            print("Création d'images de test...")
            
            # Créer une image noire de test
            test_image = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Ajouter un peu de bruit pour simuler des variations
            noise = np.random.randint(0, 30, (224, 224, 3), dtype=np.uint8)
            test_image = cv2.add(test_image, noise)
            
            # Sauvegarder plusieurs versions avec du bruit différent
            for i in range(10):
                base_image = create_test_image(state)
                # Ajouter du bruit et des variations
                noise = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
                noisy_image = np.clip(base_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                # Rotation aléatoire
                angle = np.random.randint(-15, 15)
                M = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
                rotated_image = cv2.warpAffine(noisy_image, M, (224, 224))
                
                save_path = output_dir / f'sample_{i}.jpg'
                cv2.imwrite(str(save_path), rotated_image)
                print(f"Image créée: {save_path}")
            
        except Exception as e:
            print(f"Erreur lors du traitement de {data_type}: {e}")

def main():
    print("Début de la configuration...")
    print("\nÉtape 1: Téléchargement et création des données d'exemple...")
    download_sample_data()
    
    print("\nÉtape 2: Préparation et division des données...")
    split_dataset('data/custom', 'data/processed')
    
    print("\nConfiguration terminée!")
    print("\nStructure des dossiers créée:")
    print("""
    data/
    ├── custom/
    │   ├── eyes/
    │   │   ├── open/
    │   │   └── closed/
    │   └── mouth/
    │       ├── open/
    │       └── closed/
    └── processed/
        ├── train/
        ├── val/
        └── test/
    """)
    
    print("\nVous pouvez maintenant lancer l'entraînement avec:")
    print("""
    python -m src.fatigue_detection.scripts.train \\
        --model_type emsr \\
        --data_dir data/processed \\
        --batch_size 32 \\
        --epochs 50 \\
        --output_dir models
    """)

if __name__ == '__main__':
    main() 