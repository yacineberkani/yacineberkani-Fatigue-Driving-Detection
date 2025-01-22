import tensorflow as tf
import numpy as np
from pathlib import Path

class ModelTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        
        # Métriques
        self.train_loss = tf.keras.metrics.Mean()
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.val_loss = tf.keras.metrics.Mean()
        self.val_accuracy = tf.keras.metrics.CategoricalAccuracy()
        
        # Historique
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    @tf.function
    def train_step(self, batch_x, batch_y):
        """Une étape d'entraînement"""
        with tf.GradientTape() as tape:
            predictions = self.model(batch_x, training=True)
            loss = self.loss_fn(batch_y, predictions)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(batch_y, predictions)
        
        return loss
    
    @tf.function
    def test_step(self, batch_x, batch_y):
        """Une étape de validation"""
        predictions = self.model(batch_x, training=False)
        loss = self.loss_fn(batch_y, predictions)
        
        self.val_loss.update_state(loss)
        self.val_accuracy.update_state(batch_y, predictions)
        
        return loss
        
    def train(self, train_dataset, val_dataset, epochs=50, save_dir='models'):
        """Entraînement complet du modèle"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Réinitialisation des métriques
            self.train_loss.reset_state()
            self.train_accuracy.reset_state()
            self.val_loss.reset_state()
            self.val_accuracy.reset_state()
            
            # Entraînement
            for batch_x, batch_y in train_dataset:
                self.train_step(batch_x, batch_y)
            
            # Validation
            for batch_x, batch_y in val_dataset:
                self.test_step(batch_x, batch_y)
            
            # Sauvegarde des métriques dans l'historique
            self.history['loss'].append(float(self.train_loss.result()))
            self.history['accuracy'].append(float(self.train_accuracy.result()))
            self.history['val_loss'].append(float(self.val_loss.result()))
            self.history['val_accuracy'].append(float(self.val_accuracy.result()))
            
            # Affichage des métriques
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {self.train_loss.result():.4f}")
            print(f"Train Accuracy: {self.train_accuracy.result():.4f}")
            print(f"Val Loss: {self.val_loss.result():.4f}")
            print(f"Val Accuracy: {self.val_accuracy.result():.4f}")
            
            # Early stopping et sauvegarde
            val_loss = self.val_loss.result()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_path = save_dir / 'checkpoints' / 'best_model.keras'
                save_path.parent.mkdir(exist_ok=True)
                self.model.save(save_path)
                print(f"Modèle sauvegardé dans: {save_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        
        # Sauvegarde du modèle final et de l'historique
        final_model_path = save_dir / 'final_model.keras'
        self.model.save(final_model_path)
        
        # Sauvegarde de l'historique
        history_path = save_dir / 'history.npy'
        np.save(history_path, self.history)
        
        print(f"\nModèle final sauvegardé dans: {final_model_path}")
        print(f"Historique sauvegardé dans: {history_path}")
    
    def save_model(self, path):
        """Sauvegarde le modèle"""
        if not str(path).endswith(('.keras', '.h5')):
            path = str(path) + '.keras'
        self.model.save(path)
    
    def load_model(self, path):
        """Charge le modèle"""
        self.model = tf.keras.models.load_model(path) 