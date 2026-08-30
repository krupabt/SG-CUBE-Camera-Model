import os
import sys
import unittest
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visionclaw_gui import SGCubeApp

class TestSleepWakeLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.app = SGCubeApp(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.app.on_close()

    def test_sleep_mode_resource_release(self):
        # Trigger sleep mode
        self.app.enter_sleep_mode()
        self.assertEqual(self.app.current_state, "SLEEPING")
        self.assertFalse(self.app.camera_running)
        self.assertFalse(self.app.ai_running)

    def test_wake_mode_resource_restoration(self):
        # Trigger wake / bring_to_foreground
        self.app.bring_to_foreground()
        self.assertEqual(self.app.current_state, "LISTENING")
        self.assertTrue(self.app.camera_running)

    def test_repeated_sleep_wake_cycles(self):
        for cycle in range(5):
            self.app.enter_sleep_mode()
            self.assertEqual(self.app.current_state, "SLEEPING")
            self.assertFalse(self.app.camera_running)

            self.app.bring_to_foreground()
            self.assertEqual(self.app.current_state, "LISTENING")
            self.assertTrue(self.app.camera_running)

if __name__ == "__main__":
    unittest.main()
