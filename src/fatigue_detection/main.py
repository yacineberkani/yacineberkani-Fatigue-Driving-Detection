import cv2
import numpy as np
import tensorflow as tf
from .models.improved_mtcnn import ImprovedMTCNN
from .models.e_msr_net import EMSRNet
from .metrics.fatigue_metrics import FatigueMetrics

class FatigueDetector:
    def __init__(self):
        self.mtcnn = None
        self.e_msr_net = None
        self.metrics = FatigueMetrics()
        
    def build(self):
        """Initialise et construit les modèles"""
        self.mtcnn = ImprovedMTCNN()
        self.e_msr_net = EMSRNet()
        
    def process_frame(self, frame):
        """Traite une frame vidéo"""
        # Détection du visage et des points clés
        face_box, landmarks = self.mtcnn.detect(frame)
        if face_box is None:
            return frame, False
            
        # Extraction des régions des yeux et de la bouche
        left_eye = self._extract_eye_region(frame, landmarks, 'left')
        right_eye = self._extract_eye_region(frame, landmarks, 'right')
        mouth = self._extract_mouth_region(frame, landmarks)
        
        # Prédiction des états
        left_eye_closed = self.e_msr_net.predict(left_eye) > 0.5
        right_eye_closed = self.e_msr_net.predict(right_eye) > 0.5
        mouth_open = self.e_msr_net.predict(mouth) > 0.5
        
        # Calcul de la pose de la tête
        head_pose = self._compute_head_pose(landmarks)
        is_nonfront = not self._is_front_facing(head_pose)
        
        # Mise à jour des métriques
        self.metrics.update_eye_state(left_eye_closed and right_eye_closed)
        self.metrics.update_mouth_state(mouth_open)
        self.metrics.update_head_state(is_nonfront)
        
        # Détermination de l'état de fatigue
        is_fatigued = self.metrics.is_fatigue_state()
        
        return frame, is_fatigued 