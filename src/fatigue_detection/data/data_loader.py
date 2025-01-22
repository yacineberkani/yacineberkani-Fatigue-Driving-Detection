import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
import os

class DataLoader:
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
    
    def _load_wider_face_annotations(self, split):
        """Charge les annotations WIDER FACE"""
        annotations_file = self.data_dir / 'wider_face' / 'wider_face_split' / f'wider_face_{split}_bbx_gt.txt'
        
        if not annotations_file.exists():
            raise FileNotFoundError(f"Fichier d'annotations non trouvé: {annotations_file}")
        
        image_paths = []
        labels = []
        
        try:
            with open(annotations_file, 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                if not lines[i].strip():
                    i += 1
                    continue
                
                image_name = lines[i].strip()
                if not image_name.endswith('.jpg'):
                    i += 1
                    continue
                    
                i += 1
                if i >= len(lines):
                    break
                    
                try:
                    face_count = int(lines[i].strip())
                    i += 1
                except ValueError:
                    i += 1
                    continue
                
                image_path = self.data_dir / 'wider_face' / f'WIDER_{split}' / 'images' / image_name
                
                if face_count > 0 and image_path.exists():
                    # Pour MTCNN, nous utilisons seulement la présence/absence de visage
                    image_paths.append(str(image_path))
                    labels.append([1, 0] if face_count > 0 else [0, 1])  # Format one-hot
                
                # Skip bbox annotations
                i += face_count
                    
        except Exception as e:
            print(f"Erreur lors de la lecture des annotations: {e}")
            return [], []
            
        print(f"Chargé {len(image_paths)} images avec annotations pour le split {split}")
        return image_paths, labels
    
    def create_dataset(self, split='train'):
        """Crée un tf.data.Dataset pour l'entraînement ou la validation"""
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Split {split} non valide. Utilisez 'train', 'val' ou 'test'")
        
        # Chargement des chemins et labels
        image_paths, labels = self._load_wider_face_annotations(split)
        
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

    def get_dataset_size(self, split='train'):
        """Retourne la taille du dataset"""
        image_paths, _ = self._load_wider_face_annotations(split)
        return len(image_paths) 