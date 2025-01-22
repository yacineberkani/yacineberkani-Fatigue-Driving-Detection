import tensorflow as tf
from pathlib import Path
import os

class EMSRDataLoader:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = (224, 224)
        
    def _parse_image(self, image_path, label):
        """Parse et prétraite une image"""
        # Lecture de l'image
        image_string = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        
        # Redimensionnement
        image = tf.image.resize(image, self.image_size)
        
        # Normalisation
        image = tf.cast(image, tf.float32) / 255.0
        
        return image, label
    
    def _load_data(self, split):
        """Charge les images et labels pour un split donné"""
        image_paths = []
        labels = []
        
        # Structure attendue:
        # data_dir/
        #   ├── train/
        #   │   ├── eyes/
        #   │   │   ├── open/
        #   │   │   └── closed/
        #   │   └── mouth/
        #   │       ├── open/
        #   │       └── closed/
        #   ├── val/
        #   └── test/
        
        split_dir = self.data_dir / split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Dossier non trouvé: {split_dir}")
            
        # Chargement des yeux
        eyes_open_dir = split_dir / 'eyes' / 'open'
        eyes_closed_dir = split_dir / 'eyes' / 'closed'
        
        # Chargement des images des yeux ouverts
        if eyes_open_dir.exists():
            for img_path in eyes_open_dir.glob('*.jpg'):
                image_paths.append(str(img_path))
                labels.append([1, 0])  # One-hot pour ouvert
                
        # Chargement des images des yeux fermés
        if eyes_closed_dir.exists():
            for img_path in eyes_closed_dir.glob('*.jpg'):
                image_paths.append(str(img_path))
                labels.append([0, 1])  # One-hot pour fermé
                
        print(f"Chargé {len(image_paths)} images pour le split {split}")
        return image_paths, labels
    
    def create_dataset(self, split='train'):
        """Crée un tf.data.Dataset pour l'entraînement ou la validation"""
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Split {split} non valide. Utilisez 'train', 'val' ou 'test'")
        
        # Chargement des chemins et labels
        image_paths, labels = self._load_data(split)
        
        if not image_paths:
            raise ValueError(f"Aucune donnée trouvée pour le split {split}")
        
        # Création des tenseurs
        paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        
        # Combinaison des datasets
        dataset = tf.data.Dataset.zip((paths_ds, labels_ds))
        
        # Application du parsing et prétraitement
        dataset = dataset.map(
            self._parse_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Configuration du dataset
        if split == 'train':
            dataset = dataset.shuffle(1000)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset 