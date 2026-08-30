import os
import sys
import unittest
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visionclaw_gui import SGCubeApp

class TestUICallbacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.app = SGCubeApp(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.app.on_close()

    def test_app_state_initialization(self):
        self.assertEqual(self.app.current_state, "IDLE")
        self.assertIsNotNone(self.app.engine)
        self.assertIsNotNone(self.app.active_history_session_id)

    def test_set_state_transitions(self):
        self.app.set_state("LISTENING")
        self.assertEqual(self.app.current_state, "LISTENING")
        self.app.set_state("THINKING")
        self.assertEqual(self.app.current_state, "THINKING")
        self.app.set_state("IDLE")
        self.assertEqual(self.app.current_state, "IDLE")

    def test_trigger_voice_intent(self):
        # Clear queue
        while not self.app.gui_queue.empty():
            self.app.gui_queue.get_nowait()

        # Trigger intent programmatically
        self.app._trigger_voice_intent("What is around me?")
        self.assertFalse(self.app.gui_queue.empty())

        found_transcript = False
        while not self.app.gui_queue.empty():
            item = self.app.gui_queue.get_nowait()
            if item[0] == "TRANSCRIPT_USER" and item[1] == "What is around me?":
                found_transcript = True
                break
        self.assertTrue(found_transcript)

if __name__ == "__main__":
    unittest.main()
