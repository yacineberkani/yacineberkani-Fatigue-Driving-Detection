from pathlib import Path

class Config:
    # Chemins
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    MODEL_DIR = PROJECT_ROOT / 'models'
    
    # Paramètres du modèle
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    
    # Seuils de fatigue
    EYE_THRESHOLD = 0.5
    MOUTH_THRESHOLD = 0.3
    HEAD_THRESHOLD = 0.5
    
    # Paramètres de la caméra
    CAMERA_MATRIX = None  # À définir selon votre caméra
    DIST_COEFFS = None    # À définir selon votre caméra 