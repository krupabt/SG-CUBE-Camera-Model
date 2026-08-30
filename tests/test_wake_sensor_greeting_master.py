import os
import sys
import time
import unittest
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visionclaw_gui import SGCubeApp

class TestWakeSensorGreetingMaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.app = SGCubeApp(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.app.on_close()

    def setUp(self):
        while self.app.engine.response_manager.get_next_response() is not None:
            pass
        with self.app.state_lock:
            self.app.wake_greeting_pending = True
            self.app.wake_greeting_timer = time.time()
            self.app.first_valid_frame_received = True

    def test_case_1_primary_user(self):
        self.app.engine.store.set_setting("user_name", "Alexth")
        self.app.engine.store.set_setting("user_display_name", "Alexth")

        faces = [{"name": "Alexth", "confidence": 1.0, "bbox": [10, 10, 100, 100]}]
        self.app._evaluate_wake_greeting(faces)

        self.assertFalse(self.app.wake_greeting_pending)
        res = self.app.engine.response_manager.get_next_response()
        self.assertIsNotNone(res)
        self.assertIn("Alexth", res)

    def test_case_2_known_secondary_person(self):
        self.app.engine.store.set_setting("user_name", "Alexth")
        self.app.engine.store.set_setting("user_display_name", "Alexth")

        faces = [{"name": "Rahul", "confidence": 0.95, "bbox": [10, 10, 100, 100]}]
        self.app._evaluate_wake_greeting(faces)

        self.assertFalse(self.app.wake_greeting_pending)
        res = self.app.engine.response_manager.get_next_response()
        self.assertIsNotNone(res)
        self.assertIn("Rahul", res)

    def test_case_3_unknown_person(self):
        faces = [{"name": "Unknown", "confidence": 0.0, "bbox": [10, 10, 100, 100]}]
        self.app._evaluate_wake_greeting(faces)

        self.assertFalse(self.app.wake_greeting_pending)
        res = self.app.engine.response_manager.get_next_response()
        self.assertIsNotNone(res)
        self.assertIn("Hello. I'm SG CUBE", res)

    def test_case_4_no_face_timeout(self):
        self.app.wake_greeting_timer = time.time() - 3.0
        faces = []
        self.app._evaluate_wake_greeting(faces)

        self.assertFalse(self.app.wake_greeting_pending)
        res = self.app.engine.response_manager.get_next_response()
        self.assertIsNotNone(res)

    def test_10_cycle_wake_repeatability(self):
        for cycle in range(1, 11):
            self.app.enter_sleep_mode()
            self.assertEqual(self.app.current_state, "SLEEPING")

            self.app.bring_to_foreground()
            self.assertEqual(self.app.current_state, "LISTENING")
            self.assertTrue(self.app.wake_greeting_pending)

if __name__ == "__main__":
    unittest.main()
