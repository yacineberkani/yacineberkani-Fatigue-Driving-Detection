import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelEvaluator:
    def __init__(self, model):
        self.model = model
        
    def evaluate(self, test_dataset):
        """Évalue le modèle sur le jeu de test"""
        all_predictions = []
        all_labels = []
        
        for batch_x, batch_y in test_dataset:
            predictions = self.model(batch_x, training=False)
            predictions = (predictions.numpy() > 0.5).astype(int)
            
            all_predictions.extend(predictions)
            all_labels.extend(batch_y.numpy())
            
        # Calcul des métriques
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
    def evaluate_video(self, video_path):
        """Évalue le modèle sur une vidéo complète"""
        from ..utils.video_processor import VideoProcessor
        
        video_proc = VideoProcessor(video_path)
        detector = FatigueDetector()
        detector.build()
        
        fatigue_frames = 0
        total_frames = 0
        
        for frame in video_proc.frames():
            _, is_fatigued = detector.process_frame(frame)
            if is_fatigued:
                fatigue_frames += 1
            total_frames += 1
            
        fatigue_ratio = fatigue_frames / total_frames
        return fatigue_ratio 