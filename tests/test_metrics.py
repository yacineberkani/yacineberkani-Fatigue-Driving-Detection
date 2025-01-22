import unittest
import numpy as np
from src.fatigue_detection.metrics.fatigue_metrics import FatigueMetrics

class TestFatigueMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = FatigueMetrics()
        
    def test_eye_state_update(self):
        self.metrics.update_eye_state(True)  # Yeux fermés
        self.metrics.update_eye_state(False)  # Yeux ouverts
        ecr, _, _ = self.metrics.compute_metrics()
        self.assertEqual(ecr, 0.5)
        
    def test_fatigue_state_detection(self):
        # Simuler un état de fatigue
        for _ in range(10):
            self.metrics.update_eye_state(True)
        self.assertTrue(self.metrics.is_fatigue_state())
        
    def test_metrics_reset(self):
        self.metrics.update_eye_state(True)
        self.metrics.reset()
        ecr, mor, hnfr = self.metrics.compute_metrics()
        self.assertEqual((ecr, mor, hnfr), (0.0, 0.0, 0.0)) 