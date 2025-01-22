import tensorflow as tf
from pathlib import Path
import argparse
from ..models.e_msr_net import EMSRNet
from ..models.improved_mtcnn import ImprovedMTCNN
from ..data.data_loader import DataLoader
from ..training.trainer import ModelTrainer
from ..data.emsr_data_loader import EMSRDataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Entraînement du modèle de détection de fatigue')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['mtcnn', 'emsr'],
                      help='Type de modèle à entraîner (mtcnn ou emsr)')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Chemin vers le répertoire des données')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Taille du batch')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Nombre d\'époques')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Taux d\'apprentissage')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Répertoire de sauvegarde du modèle')
    return parser.parse_args()

def train_mtcnn(args):
    """Fonction spécifique pour l'entraînement de MTCNN"""
    data_loader = DataLoader(args.data_dir, args.batch_size)
    
    # Création du modèle MTCNN
    model = ImprovedMTCNN()
    
    # Chargement des données
    train_dataset = data_loader.create_dataset('train')
    val_dataset = data_loader.create_dataset('val')
    
    # Entraînement
    trainer = ModelTrainer(model, args.learning_rate)
    trainer.train(train_dataset, val_dataset, args.epochs, args.output_dir)

def train_emsr(args):
    """Fonction spécifique pour l'entraînement de E-MSR Net"""
    data_loader = EMSRDataLoader(args.data_dir, args.batch_size)
    
    # Création du modèle E-MSR Net
    model = EMSRNet()
    
    # Chargement des données
    train_dataset = data_loader.create_dataset('train')
    val_dataset = data_loader.create_dataset('val')
    
    # Entraînement
    trainer = ModelTrainer(model, args.learning_rate)
    trainer.train(train_dataset, val_dataset, args.epochs, args.output_dir)
    
    # Sauvegarde du modèle final
    save_path = Path(args.output_dir) / 'emsr' / 'final_model.keras'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(save_path)
    print(f"\nModèle E-MSR Net sauvegardé dans: {save_path}")

def main():
    args = parse_args()
    
    if args.model_type == 'mtcnn':
        train_mtcnn(args)
    else:
        train_emsr(args)

if __name__ == '__main__':
    main() 