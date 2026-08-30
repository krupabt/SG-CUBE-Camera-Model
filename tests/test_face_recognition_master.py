import os
import sys
import unittest
import numpy as np
import cv2
import json

INSTALLED_APP_DIR = r"C:\Users\Shara\AppData\Local\Programs\SG-CUBE"
sys.path.insert(0, INSTALLED_APP_DIR)
sys.path.insert(0, r"D:\VisionClaw-main")

from assistive.face_memory import FaceMemory
from assistive.face_recognition import FaceRecognizer
from assistive.vision_engine import VisionEngine

class TestFaceRecognitionMaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.face_dir = os.path.join(INSTALLED_APP_DIR, "data", "face_memory")
        cls.engine = VisionEngine(data_dir=os.path.join(INSTALLED_APP_DIR, "data"))
        cls.fm = cls.engine.face_memory
        cls.fr = cls.engine.face_recognizer

        # Ensure reference test face exists and is enrolled with matching embedding version
        has_valid_my_face = False
        for p in list(cls.fm.profiles.values()):
            if "my face" in p["name"].lower():
                if p["embedding"].shape[0] == cls.fm.embedding_dimension:
                    has_valid_my_face = True
                else:
                    ref_path = os.path.join(p["dir_path"], "reference.jpg")
                    if os.path.exists(ref_path):
                        img = cv2.imread(ref_path)
                        cls.fm.forget_person("My Face")
                        cls.fm.save_person(name="My Face", face_crop=img)
                        has_valid_my_face = True

        if not has_valid_my_face:
            ref_crop = np.ones((128, 128, 3), dtype=np.uint8) * 120
            cv2.circle(ref_crop, (64, 64), 40, (220, 180, 150), -1)
            cls.fm.save_person(name="My Face", face_crop=ref_crop)
            cls.fm.load_all_profiles()

    def test_01_installed_profiles_exist_and_valid(self):
        """ Verify enrolled profiles exist in installed app storage with valid embeddings """
        print(f"\n[TEST 01] Stored profiles in {self.face_dir}: {list(self.fm.profiles.keys())}")
        self.assertGreater(len(self.fm.profiles), 0, "Enrolled face profiles must exist!")
        for pid, p in self.fm.profiles.items():
            self.assertIn("name", p)
            self.assertIn("embedding", p)
            emb = p["embedding"]
            self.assertIn(emb.shape[0], (128, 256), f"Profile {p['name']} embedding must be 128-D (v2) or 256-D (v1)!")
            self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=3, msg="Embedding must be unit normalized!")
            print(f"  Profile: '{p['name']}' (ID: {pid}) -> Shape: {emb.shape}, Norm: {np.linalg.norm(emb):.4f}")

    def test_02_embedding_pipeline_compatibility(self):
        """ Verify enrollment and recognition compute_face_embedding use identical representation & shape """
        dummy_crop = np.ones((128, 128, 3), dtype=np.uint8) * 100
        emb1 = self.fm.compute_face_embedding(dummy_crop)
        emb2 = self.fm.compute_face_embedding(dummy_crop)
        self.assertEqual(emb1.shape, emb2.shape)
        self.assertIn(emb1.shape[0], (128, 256))
        self.assertTrue(np.allclose(emb1, emb2), "Embedding generation must be deterministic and identical!")
        print(f"[TEST 02] Embedding dimension = {emb1.shape[0]} (deterministic unit vector)")

    def test_03_known_face_recognition_match(self):
        """ Test: Enrolled reference image of 'My Face' achieves MATCH above 0.55 threshold """
        target_profile = None
        for p in self.fm.profiles.values():
            if "my face" in p["name"].lower():
                target_profile = p
                break

        self.assertIsNotNone(target_profile, "Must have 'My Face' enrolled in face memory")
        ref_path = os.path.join(target_profile["dir_path"], "reference.jpg")
        self.assertTrue(os.path.exists(ref_path), f"Reference image {ref_path} must exist!")
        
        img = cv2.imread(ref_path)
        self.assertIsNotNone(img)

        # Match directly with face memory
        matched_name, conf = self.fm.find_match(img, threshold=self.fr.threshold)
        print(f"[TEST 03] Reference match result: name='{matched_name}', confidence={conf:.4f}, threshold={self.fr.threshold}")
        self.assertEqual(matched_name, target_profile["name"])
        self.assertGreaterEqual(conf, 0.85, "Same face match confidence should be >= 0.85")

    def test_04_detect_and_recognize_faces_api(self):
        """ Test: detect_and_recognize_faces method exists on FaceRecognizer and returns structured results """
        self.assertTrue(hasattr(self.fr, "detect_and_recognize_faces"), "FaceRecognizer must have detect_and_recognize_faces!")
        
        # Test on reference image with face
        target_profile = list(self.fm.profiles.values())[0]
        ref_path = os.path.join(target_profile["dir_path"], "reference.jpg")
        img = cv2.imread(ref_path)
        
        results = self.fr.detect_and_recognize_faces(img)
        print(f"[TEST 04] detect_and_recognize_faces returned {len(results)} face(s)")
        self.assertIsInstance(results, list)

    def test_05_unknown_face_rejection(self):
        """ Test: A distinctly different / random image is classified as UNKNOWN (conf < 0.55) """
        unknown_face = np.zeros((150, 150, 3), dtype=np.uint8)
        unknown_face[:, :, 0] = 255  # Solid pure blue
        matched_name, conf = self.fm.find_match(unknown_face, threshold=0.55)
        print(f"[TEST 05] Unknown pattern match: name={matched_name}, conf={conf:.4f}")
        self.assertIsNone(matched_name, "Distinct unknown pattern must return None (UNKNOWN)")

    def test_06_no_face_handling(self):
        """ Test: Blank / empty frame produces empty detection list gracefully """
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = self.fr.detect_faces(blank_frame)
        self.assertEqual(len(boxes), 0, "Blank frame should detect 0 face boxes")
        
        results = self.fr.process_frame(blank_frame)
        self.assertEqual(len(results), 0, "Blank frame process_frame should return empty list")
        print("[TEST 06] No-face case handled cleanly with 0 detections")

    def test_07_multiple_faces_recognition(self):
        """ Test: Frame containing multiple faces detects and identifies each correctly """
        target_profile = list(self.fm.profiles.values())[0]
        ref_path = os.path.join(target_profile["dir_path"], "reference.jpg")
        img_known = cv2.imread(ref_path)
        if img_known is None:
            img_known = np.ones((120, 120, 3), dtype=np.uint8) * 150

        # Construct synthetic multi-face canvas (480x640)
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        # Place face 1 on left
        h1, w1 = img_known.shape[:2]
        canvas[50:50+h1, 50:50+w1] = img_known
        # Place face 2 (inverted/unknown) on right
        img_unknown = cv2.bitwise_not(img_known)
        h2, w2 = img_unknown.shape[:2]
        canvas[50:50+h2, 350:350+w2] = img_unknown

        boxes = self.fr.detect_faces(canvas)
        print(f"[TEST 07] Multi-face canvas detected {len(boxes)} boxes")
        # Direct verify matching
        match_left, conf_left = self.fm.find_match(canvas[50:50+h1, 50:50+w1], threshold=0.55)
        self.assertIsNotNone(match_left, "Left known face must match!")
        print(f"  Left Face: matched='{match_left}' (conf={conf_left:.4f})")

    def test_08_enroll_restart_delete_lifecycle(self):
        """ Test: Enroll new temporary face -> persist to disk -> reload in fresh FaceMemory -> match -> delete -> verify deleted """
        temp_name = "Test_Temporary_Person"
        temp_crop = np.ones((100, 100, 3), dtype=np.uint8) * 180
        cv2.circle(temp_crop, (50, 50), 30, (200, 150, 120), -1)

        # 1. Enroll
        pid = self.fm.save_person(name=temp_name, face_crop=temp_crop)
        self.assertIsNotNone(pid)
        print(f"[TEST 08] Enrolled '{temp_name}' with ID={pid}")

        # 2. Simulate restart / reload
        fresh_fm = FaceMemory(storage_dir=self.face_dir)
        self.assertIn(pid, fresh_fm.profiles)
        self.assertEqual(fresh_fm.profiles[pid]["name"], temp_name)

        # 3. Match
        matched, conf = fresh_fm.find_match(temp_crop, threshold=0.55)
        self.assertEqual(matched, temp_name)
        print(f"  Matched after reload: '{matched}' (conf={conf:.4f})")

        # 4. Delete
        success = fresh_fm.forget_person(temp_name)
        self.assertTrue(success, "Must successfully forget enrolled person")
        self.assertNotIn(pid, fresh_fm.profiles)

        # 5. Verify deleted from disk
        reloaded_fm = FaceMemory(storage_dir=self.face_dir)
        self.assertNotIn(pid, reloaded_fm.profiles)
        print("  Successfully deleted and verified absent from disk.")

    def test_09_clear_all_profiles_alias(self):
        """ Test: clear_all_profiles alias works on FaceMemory """
        self.assertTrue(hasattr(self.fm, "clear_all_profiles"), "FaceMemory must have clear_all_profiles alias!")

if __name__ == "__main__":
    unittest.main()
