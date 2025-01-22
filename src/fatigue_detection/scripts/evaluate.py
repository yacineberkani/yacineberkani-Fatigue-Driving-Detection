import tensorflow as tf
import argparse
from pathlib import Path
from ..data.emsr_data_loader import EMSRDataLoader
from ..models.e_msr_net import EMSRNet
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def save_confusion_matrix(cm, save_path):
    """Sauvegarde la matrice de confusion en format PNG"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraies étiquettes')
    plt.xlabel('Prédictions')
    plt.savefig(save_path)
    plt.close()

def save_metrics_plot(history_path, save_path):
    """Charge l'historique et sauvegarde les courbes de loss/accuracy"""
    try:
        history = np.load(history_path, allow_pickle=True).item()
        
        plt.figure(figsize=(12, 5))
        
        # Courbe de loss
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Évolution de la Loss')
        plt.xlabel('Époque')
        plt.ylabel('Loss')
        plt.legend()
        
        # Courbe d'accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Val Accuracy')
        plt.title('Évolution de l\'Accuracy')
        plt.xlabel('Époque')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Erreur lors de la création des courbes: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='Évaluation du modèle de détection de fatigue')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Chemin vers le modèle sauvegardé')
    parser.add_argument('--test_dir', type=str, required=True,
                      help='Chemin vers le répertoire de test')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Taille du batch')
    return parser

def evaluate_model(model, test_dataset, results_dir):
    """Évalue le modèle et sauvegarde les résultats"""
    # Création du dossier de résultats
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Prédictions et vraies étiquettes
    all_predictions = []
    all_labels = []
    
    # Évaluation batch par batch
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    
    for batch_x, batch_y in test_dataset:
        predictions = model(batch_x, training=False)
        loss = tf.keras.losses.categorical_crossentropy(batch_y, predictions)
        test_loss.update_state(loss)
        test_accuracy.update_state(batch_y, predictions)
        
        all_predictions.extend(np.argmax(predictions, axis=1))
        all_labels.extend(np.argmax(batch_y, axis=1))
    
    # Préparation des résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "loss_moyenne": test_loss.result().numpy(),
        "precision": test_accuracy.result().numpy(),
        "rapport": classification_report(all_labels, all_predictions, 
                                      target_names=['Fermé', 'Ouvert']),
        "matrice_confusion": confusion_matrix(all_labels, all_predictions)
    }
    
    # Sauvegarde des résultats textuels
    results_file = results_dir / f"resultats_{timestamp}.txt"
    with open(results_file, "w") as f:
        f.write(f"Résultats de l'évaluation ({timestamp})\n")
        f.write("="*50 + "\n\n")
        f.write(f"Loss moyenne: {results['loss_moyenne']:.4f}\n")
        f.write(f"Précision: {results['precision']:.4f}\n\n")
        f.write("Rapport de classification:\n")
        f.write(results['rapport'])
        f.write("\nMatrice de confusion:\n")
        f.write(str(results['matrice_confusion']))
    
    # Sauvegarde de la matrice de confusion
    cm_file = results_dir / f"matrice_confusion_{timestamp}.png"
    save_confusion_matrix(results['matrice_confusion'], cm_file)
    
    # Sauvegarde des courbes d'apprentissage
    history_file = Path("models/history.npy")  # Assurez-vous que ce fichier existe
    if history_file.exists():
        curves_file = results_dir / f"courbes_apprentissage_{timestamp}.png"
        save_metrics_plot(history_file, curves_file)
    
    print(f"\nRésultats sauvegardés dans: {results_dir}")
    return results

def load_model(model_path):
    """Charge le modèle avec gestion des erreurs"""
    try:
        # Essai de chargement direct
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Erreur lors du chargement direct: {e}")
        print("Tentative de reconstruction du modèle...")
        
        # Reconstruction du modèle
        model = EMSRNet()
        model.build((None, 224, 224, 3))  # Construction de la forme du modèle
        
        try:
            # Chargement uniquement des poids
            model.load_weights(model_path)
            print("Poids chargés avec succès")
            return model
        except Exception as e:
            raise Exception(f"Échec du chargement des poids: {e}")

def main():
    parser = parse_args()
    
    # Ajout d'un argument pour le dossier de résultats
    parser.add_argument('--results_dir', type=str, default='resultats',
                      help='Dossier de sauvegarde des résultats')
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        model_path = model_path.with_suffix('.keras')
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: ni en .h5 ni en .keras")
    
    print(f"Chargement du modèle depuis: {model_path}")
    model = load_model(model_path)
    
    data_loader = EMSRDataLoader(args.test_dir, args.batch_size)
    test_dataset = data_loader.create_dataset('test')
    
    # Évaluation avec sauvegarde des résultats
    evaluate_model(model, test_dataset, args.results_dir)

if __name__ == '__main__':
    main() 