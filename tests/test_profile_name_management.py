import os
import sys
import shutil
import tempfile
import time
import unittest
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assistive.memory_store import MemoryStore
from assistive.memory_manager import MemoryManager
from assistive.face_memory import FaceMemory
from assistive.conversation_history import ConversationHistory
from visionclaw_gui import SGCubeApp

class TestProfileNameManagement(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sgcube_profile_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_fresh_clean_profile_state_empty_name(self):
        """ TEST 1: Fresh/clean profile state -> name is empty and first_run_completed is False """
        store = MemoryStore(base_dir=self.test_dir)
        self.assertEqual(store.get_setting("user_name"), "")
        self.assertEqual(store.get_setting("user_display_name"), "")
        self.assertFalse(store.get_setting("first_run_completed"))

    def test_02_first_run_setup_and_persistence(self):
        """ TEST 2 & 3: First-run setup -> user enters TestUser -> profile saves -> persists across restart """
        store = MemoryStore(base_dir=self.test_dir)
        self.assertFalse(store.get_setting("first_run_completed"))

        # User enters TestUser
        store.set_setting("user_name", "TestUser")
        store.set_setting("user_display_name", "TestUser")
        store.set_setting("first_run_completed", True)

        # Simulate restart
        reloaded_store = MemoryStore(base_dir=self.test_dir)
        self.assertEqual(reloaded_store.get_setting("user_name"), "TestUser")
        self.assertEqual(reloaded_store.get_setting("user_display_name"), "TestUser")
        self.assertTrue(reloaded_store.get_setting("first_run_completed"))

    def test_03_greeting_with_stored_name(self):
        """ TEST 4: Greeting with stored user name """
        root = tk.Tk()
        root.withdraw()
        try:
            app = SGCubeApp(root)
            app.engine.store.set_setting("user_name", "TestUser")
            app.engine.store.set_setting("user_display_name", "TestUser")

            while app.engine.response_manager.get_next_response() is not None:
                pass

            app.wake_greeting_pending = True
            app.first_valid_frame_received = True
            app.wake_greeting_timer = time.time() - 3.0  # timeout with no face

            app._evaluate_wake_greeting([])
            self.assertFalse(app.wake_greeting_pending)

            res = app.engine.response_manager.get_next_response()
            self.assertIsNotNone(res)
            self.assertIn("TestUser", res)
            self.assertNotIn("Alexth", res)
            app.on_close()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    def test_04_generic_greeting_when_no_name_set(self):
        """ TEST 5: Generic greeting when no profile name is set """
        root = tk.Tk()
        root.withdraw()
        try:
            app = SGCubeApp(root)
            app.engine.store.set_setting("user_name", "")
            app.engine.store.set_setting("user_display_name", "")

            while app.engine.response_manager.get_next_response() is not None:
                pass

            app.wake_greeting_pending = True
            app.first_valid_frame_received = True
            app.wake_greeting_timer = time.time() - 3.0

            app._evaluate_wake_greeting([])
            self.assertFalse(app.wake_greeting_pending)

            res = app.engine.response_manager.get_next_response()
            self.assertIsNotNone(res)
            self.assertIn("Hello. I'm SG CUBE", res)
            self.assertNotIn("Alexth", res)
            self.assertNotIn("User", res)
            app.on_close()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    def test_05_face_recognition_does_not_assume_sharath(self):
        """ Face greeting logic uses recognized face identity, not Alexth fallback """
        root = tk.Tk()
        root.withdraw()
        try:
            app = SGCubeApp(root)
            app.engine.store.set_setting("user_name", "TestUser")
            app.engine.store.set_setting("user_display_name", "TestUser")

            while app.engine.response_manager.get_next_response() is not None:
                pass

            # Secondary person face recognized
            app.wake_greeting_pending = True
            app.first_valid_frame_received = True
            app.wake_greeting_timer = time.time()

            faces = [{"name": "Priya", "confidence": 0.95, "bbox": [10, 10, 100, 100]}]
            app._evaluate_wake_greeting(faces)
            self.assertFalse(app.wake_greeting_pending)

            res = app.engine.response_manager.get_next_response()
            self.assertIsNotNone(res)
            self.assertIn("Priya", res)
            self.assertNotIn("Alexth", res)
            app.on_close()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

if __name__ == "__main__":
    unittest.main()
