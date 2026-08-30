import os
import shutil
import unittest
import numpy as np
import cv2
from assistive.face_memory import FaceMemory

class TestFaceMemory(unittest.TestCase):
    def setUp(self):
        self.test_dir = "data/test_face_memory_tmp"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
        self.memory = FaceMemory(storage_dir=self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_find_person(self):
        # Create dummy face crop image (synthetic face pattern)
        face_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(face_img, (50, 50), 30, (200, 150, 100), -1)
        cv2.circle(face_img, (40, 40), 5, (255, 255, 255), -1)
        cv2.circle(face_img, (60, 40), 5, (255, 255, 255), -1)

        pid = self.memory.save_person("Rahul", face_img)
        self.assertTrue(pid.startswith("rahul_"))

        people = self.memory.list_people()
        self.assertIn("Rahul", people)

        # Match exact same face crop
        match_name, score = self.memory.find_match(face_img, threshold=0.50)
        self.assertEqual(match_name, "Rahul")
        self.assertGreaterEqual(score, 0.50)

    def test_forget_person(self):
        face_img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.memory.save_person("Sahana", face_img)
        self.assertIn("Sahana", self.memory.list_people())

        success = self.memory.forget_person("Sahana")
        self.assertTrue(success)
        self.assertNotIn("Sahana", self.memory.list_people())

if __name__ == "__main__":
    unittest.main()
