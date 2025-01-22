import cv2
import numpy as np
from pathlib import Path
import albumentations as A

def prepare_dataset(input_dir, output_dir, augment=True):
    """Prépare et augmente le dataset"""
    
    # Définir les transformations pour l'augmentation
    transform = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Resize(224, 224),
        A.Normalize()
    ])
    
    # Créer les répertoires
    output_dir = Path(output_dir)
    for split in ['train', 'val', 'test']:
        for part in ['eyes', 'mouth']:
            for state in ['open', 'closed']:
                (output_dir / split / part / state).mkdir(parents=True, exist_ok=True)
    
    # Traiter les images
    input_dir = Path(input_dir)
    for img_path in input_dir.rglob('*.jpg'):
        # Lire l'image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Appliquer les transformations
        transformed = transform(image=img)
        processed_img = transformed['image']
        
        # Sauvegarder l'image
        relative_path = img_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        cv2.imwrite(str(output_path), cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
        
        # Augmentation si nécessaire
        if augment:
            for i in range(3):  # Créer 3 versions augmentées
                transformed = transform(image=img)
                aug_img = transformed['image']
                aug_path = output_path.parent / f"{output_path.stem}_aug{i}{output_path.suffix}"
                cv2.imwrite(str(aug_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)) 