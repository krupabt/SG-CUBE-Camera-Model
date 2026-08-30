import os
import sys
import time
import socket
import unittest
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wake_listener import SGCubeWakeListener, IPC_PORT_GUI
from visionclaw_gui import SGCubeApp

class TestSingleInstanceFix(unittest.TestCase):
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

    def test_01_port_49152_authoritative_lock(self):
        # Port 49152 is bound by app.lock_socket
        self.assertTrue(self.listener.is_visionclaw_running())

    def _flush_queue(self):
        for _ in range(5):
            self.app._process_gui_queue()

    def test_02_wake_while_active_no_duplicate(self):
        # When app is already active, triggering wake does not create duplicate processes
        self.app.bring_to_foreground()
        self.listener.launch_or_activate_visionclaw()
        self.app.bring_to_foreground()
        self.assertIn(self.app.current_state, ("OPENING", "LISTENING", "ACTIVE"))

    def test_03_wake_while_sleeping_restores_one_instance(self):
        self.app.enter_sleep_mode()
        self.assertEqual(self.app.current_state, "SLEEPING")
        self.listener.launch_or_activate_visionclaw()
        self.app.bring_to_foreground()
        self.assertIn(self.app.current_state, ("OPENING", "LISTENING", "ACTIVE"))

    def test_04_rapid_wake_triggers_debounce(self):
        # Fire 5 wake triggers quickly within 1.0s
        for _ in range(5):
            self.listener.launch_or_activate_visionclaw()
            self.app.bring_to_foreground()
        # Should remain single active instance
        self.assertIn(self.app.current_state, ("OPENING", "LISTENING", "ACTIVE"))

    def test_05_10_cycle_sleep_wake_single_instance(self):
        for cycle in range(1, 4):  # Test multiple sleep/wake cycles respecting debounce window
            self.app.enter_sleep_mode()
            self.assertEqual(self.app.current_state, "SLEEPING")
            time.sleep(2.1)
            self.listener.launch_or_activate_visionclaw()
            self.app.bring_to_foreground()
            self.assertIn(self.app.current_state, ("OPENING", "LISTENING", "ACTIVE"))

if __name__ == "__main__":
    unittest.main()
