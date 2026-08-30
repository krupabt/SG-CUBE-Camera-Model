import os
import sys
import time
import unittest
import numpy as np
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assistive.vision_engine import VisionEngine
from visionclaw_gui import SGCubeApp

class TestCameraServiceLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.app = SGCubeApp(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.app.on_close()

    def setUp(self):
        self.engine = VisionEngine(data_dir="data/test_camera_tmp")

    def test_empty_scene_frame_processing(self):
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        info = self.engine.process_frame(empty_frame)

        self.assertIsNotNone(info)
        self.assertEqual(len(info.get("faces", [])), 0)
        self.assertIn("safety", info)

    def test_unknown_face_frame_processing(self):
        face_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        import cv2
        cv2.rectangle(face_frame, (100, 100), (300, 350), (100, 150, 180), -1)

        info = self.engine.process_frame(face_frame)
        self.assertIsNotNone(info)

    def test_camera_independent_from_ai_state(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            res = self.engine.process_frame(frame)
            self.assertIsNotNone(res)

    def test_camera_start_and_first_frame(self):
        self.app.start_camera()
        self.assertTrue(self.app.camera_running)
        self.assertIsNotNone(self.app.camera_thread)

    def test_sleep_releases_camera(self):
        self.app.start_camera()
        time.sleep(0.1)
        self.app.stop_camera()
        self.assertFalse(self.app.camera_running)
        self.assertIsNone(self.app.camera_thread)

    def test_wake_restarts_camera(self):
        self.app.stop_camera()
        self.app.bring_to_foreground()
        self.assertTrue(self.app.camera_running)
        self.assertIsNotNone(self.app.camera_thread)

    def test_5_cycle_sleep_wake_camera_regression(self):
        for cycle in range(1, 6):
            # Sleep -> Camera OFF
            self.app.enter_sleep_mode()
            self.assertFalse(self.app.camera_running)
            self.assertIsNone(self.app.camera_thread)

            # Wake -> Camera ON
            self.app.bring_to_foreground()
            self.assertTrue(self.app.camera_running)
            self.assertIsNotNone(self.app.camera_thread)
            print(f"[TEST] 5-cycle sleep/wake camera test iteration {cycle}/5 PASS")

    def test_3_cycle_close_reopen_camera_regression(self):
        for cycle in range(1, 4):
            # Close -> Camera Teardown
            self.app.stop_camera()
            self.assertFalse(self.app.camera_running)

            # Reopen -> Camera Start
            self.app.start_camera()
            self.assertTrue(self.app.camera_running)
            print(f"[TEST] 3-cycle close/reopen camera test iteration {cycle}/3 PASS")

    def test_duplicate_camera_worker_protection(self):
        self.app.start_camera()
        t1 = self.app.camera_thread
        # Calling start_camera while running must not spawn duplicate thread
        self.app.start_camera()
        t2 = self.app.camera_thread
        self.assertEqual(t1, t2)

if __name__ == "__main__":
    unittest.main()
