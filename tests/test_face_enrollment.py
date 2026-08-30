import os
import sys
import shutil
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assistive.face_memory import FaceMemory
from assistive.face_recognition import FaceRecognizer

class TestFaceEnrollment(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.abspath("data/test_face_enrollment_db")
        os.makedirs(self.test_dir, exist_ok=True)
        self.face_memory = FaceMemory(storage_dir=self.test_dir)
        self.recognizer = FaceRecognizer(face_memory=self.face_memory)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_face_enrollment_pipeline(self):
        # Create synthetic frame with face matching skin-tone filter & aspect ratio
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(frame, (80, 50), (220, 250), (100, 150, 180), -1)

        res = self.recognizer.enroll_active_face(frame, "Sahana")
        self.assertTrue(res["success"])
        self.assertEqual(res["name"], "Sahana")

        # Verify profile is in FaceMemory
        people = self.face_memory.list_people()
        self.assertIn("Sahana", people)

if __name__ == "__main__":
    unittest.main()
