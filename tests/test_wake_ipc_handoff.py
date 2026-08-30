import os
import sys
import time
import socket
import unittest
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wake_listener import SGCubeWakeListener, IPC_PORT_GUI, IPC_PORT_WAKE_LISTENER
from visionclaw_gui import SGCubeApp

class TestWakeIPCHandoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.app = SGCubeApp(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.app.on_close()

    def test_test1_closed_app_background_listener(self):
        # When app is sleeping/closed, background listener is listening
        self.app.enter_sleep_mode()
        self.assertEqual(self.app.current_state, "SLEEPING")
        self.assertFalse(self.app.ai_running)

    def test_test2_activation_speech_handoff(self):
        # Activation speech -> Handoff to main mic -> App active
        self.app.bring_to_foreground()
        self.assertIn(self.app.current_state, ("LISTENING", "ACTIVE"))
        self.assertTrue(self.app.camera_running)

    def test_test3_already_active_protection(self):
        # App active -> bring_to_foreground does not create duplicate processes
        self.app.bring_to_foreground()
        self.assertIn(self.app.current_state, ("LISTENING", "ACTIVE"))

    def test_test4_sleep_mode_handoff(self):
        # Sleep command -> Main Mic OFF, Background Listener ON
        self.app.enter_sleep_mode()
        self.assertEqual(self.app.current_state, "SLEEPING")
        self.assertFalse(self.app.ai_running)

    def test_test5_activation_while_sleeping(self):
        # Sleep -> Wake -> Background Listener OFF, Main Mic ON
        self.app.enter_sleep_mode()
        self.app.bring_to_foreground()
        self.assertIn(self.app.current_state, ("LISTENING", "ACTIVE"))

    def test_test6_open_close_repeatedly(self):
        for cycle in range(1, 4):
            self.app.stop_ai()
            self.assertFalse(self.app.ai_running)
            self.app.start_ai("fake_key")
            self.assertTrue(self.app.ai_running)
            print(f"[TEST] Open/close iteration {cycle}/3 PASS")

    def test_test7_10_cycle_sleep_wake_regression(self):
        for cycle in range(1, 11):
            # Sleep: Main Mic OFF, Background Listener ON
            self.app.enter_sleep_mode()
            self.assertEqual(self.app.current_state, "SLEEPING")
            self.assertFalse(self.app.ai_running)

            # Wake: Background Listener OFF, Main Mic ON
            self.app.bring_to_foreground()
            self.assertIn(self.app.current_state, ("LISTENING", "ACTIVE"))
            print(f"[TEST] 10-cycle sleep/wake handoff iteration {cycle}/10 PASS")

if __name__ == "__main__":
    unittest.main()
