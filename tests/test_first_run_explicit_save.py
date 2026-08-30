import os
import sys
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assistive.memory_store import MemoryStore
from assistive.memory_manager import MemoryManager
from assistive.face_memory import FaceMemory
from assistive.command_router import CommandRouter
from assistive.vision_engine import VisionEngine

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "first_run_test_data")

class TestFirstRunExplicitSave(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        cls.engine = VisionEngine(data_dir=TEST_DATA_DIR)

    def test_01_first_run_detection_and_completion(self):
        # Set fresh install state
        self.engine.store.set_setting("first_run_completed", False)
        self.assertFalse(self.engine.store.get_setting("first_run_completed"))

        # Perform onboarding profile setup
        self.engine.store.set_setting("user_name", "Alexth")
        self.engine.store.set_setting("user_display_name", "Alexth")
        self.engine.store.set_setting("first_run_completed", True)

        # Reload store to simulate restart
        reloaded_store = MemoryStore(base_dir=TEST_DATA_DIR)
        self.assertTrue(reloaded_store.get_setting("first_run_completed"))
        self.assertEqual(reloaded_store.get_setting("user_display_name"), "Alexth")

    def test_02_explicit_memory_save_and_confirmation(self):
        # 1. Explicit save request
        resp1 = self.engine.process_user_speech_query("Remember that my favorite color is blue.")
        self.assertIn("Got it. I will remember that", resp1)

        # 2. Verify stored in memories.db
        recalled = self.engine.memory.recall_memory("what is my favorite color?")
        self.assertIsNotNone(recalled)
        self.assertIn("blue", recalled.lower())

        # 3. Standard statement (no explicit save keyword) should not be saved as intent
        route = self.engine.router.route_intent("I like blue.")
        self.assertEqual(route["intent"], "GENERAL")

        # 4. Reload engine to test persistence across restart
        reloaded_engine = VisionEngine(data_dir=TEST_DATA_DIR)
        recalled_after_restart = reloaded_engine.memory.recall_memory("favorite color")
        self.assertIsNotNone(recalled_after_restart)
        self.assertIn("blue", recalled_after_restart.lower())

    def test_03_explicit_face_save_and_confirmation(self):
        # Create synthetic face crop frame
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (250, 250), (120, 150, 200), -1)
        self.engine.current_frame = frame

        # Route face save command
        resp = self.engine.process_user_speech_query("Remember this person as Rahul")
        self.assertIn("remembered this face as Rahul", resp)

        # Reload face memory to verify persistence across restart
        reloaded_engine = VisionEngine(data_dir=TEST_DATA_DIR)
        self.assertIn("Rahul", reloaded_engine.face_memory.list_people())

    def test_04_face_save_error_when_no_face_present(self):
        # Set empty frame (no face)
        self.engine.current_frame = None

        # Route face save command
        resp = self.engine.process_user_speech_query("Remember this person as Alex")
        self.assertTrue(any(w in resp.lower() for w in ["no visual frame", "can't clearly see", "no face"]))

    def test_05_data_separation_verification(self):
        # Profile, Memory, Face, History paths must be isolated
        self.assertEqual(self.engine.store.pref_dir, os.path.join(TEST_DATA_DIR, "user_preferences"))
        self.assertEqual(self.engine.face_memory.storage_dir, os.path.join(TEST_DATA_DIR, "face_memory"))
        self.assertEqual(self.engine.memory.db_dir, os.path.join(TEST_DATA_DIR, "memory"))
        self.assertEqual(self.engine.history.db_dir, os.path.join(TEST_DATA_DIR, "history"))

if __name__ == "__main__":
    unittest.main()
