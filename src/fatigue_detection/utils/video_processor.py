import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
            
    def frames(self):
        """Générateur de frames"""
        if not self.cap:
            self.cap = cv2.VideoCapture(self.video_path)
            
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
            
    def process_video(self, output_path, process_fn):
        """Traite la vidéo frame par frame"""
        with self:
            # Obtenir les propriétés de la vidéo
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            # Créer l'objet VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Traiter chaque frame
            for frame in self.frames():
                processed_frame = process_fn(frame)
                out.write(processed_frame)
                
            out.release() 