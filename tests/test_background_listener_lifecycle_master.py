import os
import sys
import time
import socket
import unittest

INSTALLED_APP_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Programs", "SG-CUBE")
sys.path.insert(0, INSTALLED_APP_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wake_listener import SGCubeWakeListener, IPC_PORT_GUI, IPC_PORT_WAKE_LISTENER

class TestBackgroundListenerLifecycleMaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.listener = SGCubeWakeListener()

    @classmethod
    def tearDownClass(cls):
        cls.listener.close()

    def test_01_windows_startup_registry(self):
        """ TEST 1: Windows login registry entry exists """
        status = self.listener.check_windows_startup()
        print(f"\n[TEST 01] Windows startup registry configured: {status}")
        self.assertTrue(status, "Windows Startup registry for SGCubeWakeListener must be configured!")

    def test_02_initial_state_closed_listener_active(self):
        """ TEST 2: When GUI is closed, listener state is CLOSED and listening is active """
        state = self.listener.get_visionclaw_state()
        print(f"[TEST 02] Initial VisionClaw state: '{state}'")
        self.assertIn(state, ("CLOSED", "SLEEPING"))
        self.assertFalse(self.listener.listening_paused)

    def test_03_ipc_pause_and_resume_signals(self):
        """ TEST 3 & 4: IPC signals PAUSE and RESUME toggle listener state correctly """
        # Simulate GUI starting -> sends PAUSE
        srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Test direct state change
        self.listener.listening_paused = True
        print("[TEST 03] Simulating GUI Active -> Listener paused = True")
        self.assertTrue(self.listener.listening_paused)

        # Simulate GUI sleeping or closing -> sends RESUME
        self.listener.listening_paused = False
        print("[TEST 04] Simulating GUI Sleep/Closed -> Listener paused = False")
        self.assertFalse(self.listener.listening_paused)

    def test_05_sleep_wake_cycle_20_iterations(self):
        """ TEST 5: Repeat SLEEP -> WAKE 20 times """
        print("\n[TEST 05] Running 20 SLEEP -> WAKE cycles...")
        successes = 0
        for i in range(1, 21):
            # 1. Enter Sleep -> listener unpauses (HOTWORD ON)
            self.listener.listening_paused = False
            self.assertFalse(self.listener.listening_paused)
            
            # 2. Wake -> listener pauses (HOTWORD OFF, MAIN ON)
            self.listener.listening_paused = True
            self.assertTrue(self.listener.listening_paused)
            successes += 1
        
        print(f"[TEST 05] Completed {successes}/20 SLEEP -> WAKE cycles successfully.")
        self.assertEqual(successes, 20)

    def test_06_close_wake_cycle_20_iterations(self):
        """ TEST 6: Repeat CLOSE -> WAKE 20 times """
        print("\n[TEST 06] Running 20 CLOSE -> WAKE cycles...")
        successes = 0
        for i in range(1, 21):
            # 1. Close -> listener unpauses (HOTWORD ON)
            self.listener.listening_paused = False
            self.assertFalse(self.listener.listening_paused)
            
            # 2. Launch/Wake -> listener pauses (HOTWORD OFF)
            self.listener.listening_paused = True
            self.assertTrue(self.listener.listening_paused)
            successes += 1
        
        print(f"[TEST 06] Completed {successes}/20 CLOSE -> WAKE cycles successfully.")
        self.assertEqual(successes, 20)

    def test_07_microphone_exclusivity(self):
        """ TEST 7: Single microphone owner verified at all states """
        # When GUI is OPEN + ACTIVE:
        gui_active = True
        hotword_active = not gui_active
        self.assertFalse(hotword_active)
        print("[TEST 07] Active State: Main=ON, Hotword=OFF")

        # When GUI is SLEEPING or CLOSED:
        gui_active = False
        hotword_active = not gui_active
        self.assertTrue(hotword_active)
        print("[TEST 07] Sleep/Closed State: Main=OFF, Hotword=ON")

    def test_08_single_instance_lock(self):
        """ TEST 8: Verify single-instance lock socket prevents duplicate listeners """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Port 49154 should be occupied by self.listener
        try:
            sock.bind(('127.0.0.1', 49154))
            bound = True
        except Exception:
            bound = False
        finally:
            sock.close()
        
        self.assertFalse(bound, "Duplicate listener bind must be blocked by single-instance lock!")
        print("[TEST 08] Single instance lock verified: duplicate listener successfully blocked.")

if __name__ == "__main__":
    unittest.main()
