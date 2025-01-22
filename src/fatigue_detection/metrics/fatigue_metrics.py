import numpy as np

class FatigueMetrics:
    def __init__(self):
        # Seuils de fatigue
        self.eye_threshold = 0.5    # ECR threshold
        self.mouth_threshold = 0.3  # MOR threshold  
        self.head_threshold = 0.5   # HNFR threshold
        
        # Compteurs de frames
        self.total_frames = 0
        self.closed_eyes_frames = 0
        self.open_mouth_frames = 0
        self.head_nonfront_frames = 0
        
    def update_eye_state(self, is_closed):
        """Mise à jour du compteur des yeux fermés"""
        self.total_frames += 1
        if is_closed:
            self.closed_eyes_frames += 1
            
    def update_mouth_state(self, is_open):
        """Mise à jour du compteur de bouche ouverte"""
        if is_open:
            self.open_mouth_frames += 1
            
    def update_head_state(self, is_nonfront):
        """Mise à jour du compteur de tête non frontale"""
        if is_nonfront:
            self.head_nonfront_frames += 1
            
    def compute_metrics(self):
        """Calcul des métriques de fatigue"""
        if self.total_frames == 0:
            return 0.0, 0.0, 0.0
            
        ecr = self.closed_eyes_frames / self.total_frames
        mor = self.open_mouth_frames / self.total_frames
        hnfr = self.head_nonfront_frames / self.total_frames
        
        return ecr, mor, hnfr
        
    def is_fatigue_state(self):
        """Détermine l'état de fatigue basé sur la fusion des indices"""
        ecr, mor, hnfr = self.compute_metrics()
        
        # Selon l'équation (8) de l'article
        if (ecr >= self.eye_threshold or 
            mor >= self.mouth_threshold or 
            hnfr >= self.head_threshold):
            return True
        return False
        
    def reset(self):
        """Réinitialise les compteurs"""
        self.total_frames = 0
        self.closed_eyes_frames = 0
        self.open_mouth_frames = 0
        self.head_nonfront_frames = 0 