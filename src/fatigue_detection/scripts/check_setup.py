import sys
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path

def check_setup():
    """Vérifie que tout est correctement configuré"""
    print("Vérification de la configuration...")
    
    # 1. Vérification des versions
    print("\n1. Versions des bibliothèques:")
    print(f"Python: {sys.version}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")
    
    # 2. Vérification des dossiers
    print("\n2. Structure des dossiers:")
    project_root = Path(__file__).parent.parent.parent.parent
    required_dirs = [
        'data',
        'data/custom',
        'data/processed',
        'models',
        'logs'
    ]
    
    for dir_path in required_dirs:
        path = project_root / dir_path
        exists = path.exists()
        print(f"{dir_path}: {'✓' if exists else '✗'}")
        
        if not exists:
            path.mkdir(parents=True)
            print(f"  -> Créé {dir_path}")
    
    # 3. Vérification GPU
    print("\n3. Configuration GPU:")
    if tf.config.list_physical_devices('GPU'):
        print("GPU disponible ✓")
    else:
        print("Pas de GPU détecté (CPU sera utilisé)")
    
    # 4. Vérification des données
    print("\n4. Vérification des données:")
    data_dirs = [
        'data/custom/eyes/open',
        'data/custom/eyes/closed',
        'data/custom/mouth/open',
        'data/custom/mouth/closed'
    ]
    
    for dir_path in data_dirs:
        path = project_root / dir_path
        nb_images = len(list(path.glob('*.jpg'))) if path.exists() else 0
        print(f"{dir_path}: {nb_images} images")

if __name__ == '__main__':
    check_setup() 