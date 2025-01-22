import shutil
from pathlib import Path
import random

def create_directory_structure(base_dir):
    """Crée la structure de dossiers pour E-MSR Net"""
    base_dir = Path(base_dir)
    
    # Structure des dossiers
    splits = ['train', 'val', 'test']
    categories = ['eyes', 'mouth']
    states = ['open', 'closed']
    
    # Création des dossiers
    for split in splits:
        for category in categories:
            for state in states:
                dir_path = base_dir / split / category / state
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"Créé: {dir_path}")

def split_dataset(source_dir, processed_dir, train_ratio=0.7, val_ratio=0.15):
    """Répartit les images entre train, val et test"""
    source_dir = Path(source_dir)
    processed_dir = Path(processed_dir)
    
    # Création de la structure
    create_directory_structure(processed_dir)
    
    # Pour les yeux et la bouche
    for category in ['eyes', 'mouth']:
        for state in ['open', 'closed']:
            # Source des images
            src_path = source_dir / category / state
            if not src_path.exists():
                print(f"Dossier source non trouvé: {src_path}")
                continue
                
            # Liste toutes les images
            images = list(src_path.glob('*.jpg'))
            random.shuffle(images)
            
            # Calcul des indices de split
            n_images = len(images)
            n_train = int(n_images * train_ratio)
            n_val = int(n_images * val_ratio)
            
            # Répartition des images
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Copie des images
            for img in train_images:
                dest = processed_dir / 'train' / category / state / img.name
                shutil.copy2(img, dest)
                
            for img in val_images:
                dest = processed_dir / 'val' / category / state / img.name
                shutil.copy2(img, dest)
                
            for img in test_images:
                dest = processed_dir / 'test' / category / state / img.name
                shutil.copy2(img, dest)
                
            print(f"Catégorie {category}/{state}:")
            print(f"  Train: {len(train_images)} images")
            print(f"  Val: {len(val_images)} images")
            print(f"  Test: {len(test_images)} images")

def main():
    # Chemins des dossiers
    source_dir = Path('data/custom')  # Dossier contenant vos images brutes
    processed_dir = Path('data/processed')  # Dossier de destination
    
    # Création de la structure et répartition des données
    split_dataset(source_dir, processed_dir)
    
    print("\nPréparation des données terminée!")
    print(f"Les données traitées sont dans: {processed_dir}")

if __name__ == '__main__':
    main() 