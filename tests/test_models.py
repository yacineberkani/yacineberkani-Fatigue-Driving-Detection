import unittest
import tensorflow as tf
import numpy as np
from src.fatigue_detection.models.e_msr_net import EMSRNet
from src.fatigue_detection.models.improved_mtcnn import ImprovedMTCNN

class TestEMSRNet(unittest.TestCase):
    def setUp(self):
        self.model = EMSRNet()
        self.input_shape = (224, 224, 3)
        
    def test_model_output_shape(self):
        batch_size = 1
        input_tensor = tf.random.normal((batch_size,) + self.input_shape)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (batch_size, 2))
        
    def test_model_prediction_range(self):
        batch_size = 4
        input_tensor = tf.random.normal((batch_size,) + self.input_shape)
        predictions = self.model(input_tensor)
        self.assertTrue(tf.reduce_all(predictions >= 0))
        self.assertTrue(tf.reduce_all(predictions <= 1))

class TestImprovedMTCNN(unittest.TestCase):
    def setUp(self):
        self.mtcnn = ImprovedMTCNN()
        
    def test_face_detection(self):
        # Créer une image test avec un visage
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        face_box, landmarks = self.mtcnn.detect(image)
        self.assertIsNotNone(face_box)
        self.assertEqual(len(landmarks), 5)  # 5 points clés 