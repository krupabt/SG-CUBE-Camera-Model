import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assistive.color_detector import ColorDetector

class TestColorDetector(unittest.TestCase):

    def setUp(self):
        self.detector = ColorDetector()

    def test_ambient_light_bright(self):
        # White image (high luminance)
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        res = self.detector.check_ambient_light(frame)
        self.assertEqual(res["light_level"], "BRIGHT")

    def test_ambient_light_dark(self):
        # Black image (low luminance)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        res = self.detector.check_ambient_light(frame)
        self.assertEqual(res["light_level"], "DARK")

    def test_dominant_color_red(self):
        # Pure Red frame in BGR format -> (0, 0, 255)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = (0, 0, 255)
        res = self.detector.detect_dominant_color(frame)
        self.assertEqual(res["color_name"], "Red")

    def test_dominant_color_blue(self):
        # Pure Blue frame in BGR format -> (255, 0, 0)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)
        res = self.detector.detect_dominant_color(frame)
        self.assertIn("Blue", res["color_name"])

if __name__ == "__main__":
    unittest.main()
