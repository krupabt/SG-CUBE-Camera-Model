import os
import sys
import time
import unittest
import numpy as np
import cv2
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visionclaw_gui import SGCubeApp
from wake_listener import SGCubeWakeListener

class TestSleepWakeGreetings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.app = SGCubeApp(cls.root)
        cls.listener = SGCubeWakeListener()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'listener') and cls.listener:
            cls.listener.close()
        cls.app.on_close()

    def _get_latest_assistive_message(self):
        msgs = []
        while not self.app.gui_queue.empty():
            try:
                mtype, payload = self.app.gui_queue.get_nowait()
                if mtype in ("TRANSCRIPT_ASSISTIVE", "TRANSCRIPT_USER"):
                    msgs.append(payload)
            except Exception:
                break
        return msgs[-1] if msgs else None

    def test_01_wake_greeting_primary_user(self):
        self.app.engine.store.set_setting("user_name", "Alexth")
        self.app.engine.store.set_setting("user_display_name", "Alexth")

        # Simulate face detection matching primary user
        faces = [{"bbox": (10, 10, 50, 50), "name": "Alexth", "confidence": 1.0}]
        self.app.wake_greeting_pending = True
        self.app.first_valid_frame_received = True
        self.app.wake_greeting_timer = time.time()
        self.app._evaluate_wake_greeting(faces)

        msg = self._get_latest_assistive_message()
        self.assertIsNotNone(msg)
        self.assertIn("Alexth", msg)
        self.assertFalse(self.app.wake_greeting_pending)

    def test_02_wake_greeting_secondary_enrolled_person(self):
        faces = [{"bbox": (10, 10, 50, 50), "name": "Rahul", "confidence": 0.95}]
        self.app.wake_greeting_pending = True
        self.app.first_valid_frame_received = True
        self.app.wake_greeting_timer = time.time()
        self.app._evaluate_wake_greeting(faces)

        msg = self._get_latest_assistive_message()
        self.assertIsNotNone(msg)
        self.assertIn("Rahul", msg)

    def test_03_wake_greeting_unknown_person(self):
        faces = [{"bbox": (10, 10, 50, 50), "name": "Unknown", "confidence": 0.0}]
        self.app.wake_greeting_pending = True
        self.app.first_valid_frame_received = True
        self.app.wake_greeting_timer = time.time()
        self.app._evaluate_wake_greeting(faces)

        msg = self._get_latest_assistive_message()
        self.assertIsNotNone(msg)
        self.assertIn("SG CUBE", msg)

    def test_04_wake_greeting_no_face_timeout(self):
        self.app.wake_greeting_pending = True
        self.app.first_valid_frame_received = False
        self.app.wake_greeting_timer = time.time() - 3.0  # 3.0s > 2.5s timeout
        self.app._evaluate_wake_greeting([])

        msg = self._get_latest_assistive_message()
        self.assertIsNotNone(msg)
        self.assertFalse(self.app.wake_greeting_pending)

    def test_05_sleep_farewell_greeting(self):
        self.app.enter_sleep_mode()
        self.assertEqual(self.app.current_state, "SLEEPING")
        farewell = getattr(self.app, "last_farewell_msg", "")
        self.assertIn("going to sleep", farewell.lower())

    def test_06_10_cycle_sleep_wake_greetings_no_overlap(self):
        for cycle in range(1, 11):
            # 1. Wake
            self.app.bring_to_foreground()
            self.assertIn(self.app.current_state, ("OPENING", "LISTENING", "ACTIVE"))
            self.assertTrue(self.app.wake_greeting_pending)

            # Evaluate wake greeting (simulate timeout or frame)
            self.app.first_valid_frame_received = True
            self.app.wake_greeting_timer = time.time() - 3.0
            self.app._evaluate_wake_greeting([])
            self.assertFalse(self.app.wake_greeting_pending)

            # 2. Sleep
            self.app.enter_sleep_mode()
            self.assertEqual(self.app.current_state, "SLEEPING")
            farewell = getattr(self.app, "last_farewell_msg", "")
            self.assertIn("going to sleep", farewell.lower())

if __name__ == "__main__":
    unittest.main()
