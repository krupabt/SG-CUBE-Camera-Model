import unittest
import os
import shutil
import tempfile
import time
import tkinter as tk
import numpy as np

from visionclaw_gui import SGCubeApp
from assistive.vision_engine import VisionEngine
from assistive.memory_manager import MemoryManager


class TestSaveGUILifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="sgcube_gui_save_")
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.app = SGCubeApp(cls.root)
        cls.app.engine = VisionEngine(data_dir=cls.test_dir)
        cls.app.active_history_session_id = cls.app.engine.history.create_session()
        cls.app.ai_running = True
        cls.app.set_state("LISTENING")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.app.on_close()
        except Exception:
            pass
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_gui_save_returns_to_listening(self):
        """ Verify GUI returns to LISTENING state and does not freeze on 'Remember my favorite color is blue' """
        self.assertEqual(self.app.current_state, "LISTENING")

        text = "Remember my favorite color is blue"
        self.app.set_state("USER_SPEAKING")
        local_response = self.app.engine.process_user_speech_query(text)
        self.assertIsNotNone(local_response)
        self.app.gui_queue.put(("TRANSCRIPT_ASSISTIVE", local_response))

        with self.app.session_lock:
            self.app.pending_speech_prompt = f"Speak this exact response out loud: '{local_response}'"

        self.app.set_state("LISTENING")
        self.assertEqual(self.app.current_state, "LISTENING")

    def test_02_gui_save_this_contextual_conversation(self):
        """ Verify 'Save this' captures previous turn in conversation """
        self.app.engine.history.add_message(self.app.active_history_session_id, "user", "My doctor appointment is on Monday at 9am")
        self.app.engine.history.add_message(self.app.active_history_session_id, "assistant", "I noted that your doctor appointment is on Monday at 9am.")

        resp = self.app.engine.process_user_speech_query("Save this")
        self.assertIsNotNone(resp)
        self.assertIn("monday", resp.lower())

        recalled = self.app.engine.memory.recall_memory("doctor appointment")
        self.assertIsNotNone(recalled)
        self.assertIn("monday", recalled.lower())

    def test_03_gui_save_after_sleep_and_wake(self):
        """ Verify Save works seamlessly after sleep and wake cycle """
        self.app.enter_sleep_mode()
        self.assertEqual(self.app.current_state, "SLEEPING")

        self.app.bring_to_foreground()
        self.app.set_state("LISTENING")
        self.assertEqual(self.app.current_state, "LISTENING")

        resp = self.app.engine.process_user_speech_query("Remember that my brother name is Rohan")
        self.assertIsNotNone(resp)
        self.assertIn("rohan", resp.lower())

        recalled = self.app.engine.memory.recall_memory("brother")
        self.assertIsNotNone(recalled)
        self.assertIn("rohan", recalled.lower())


if __name__ == "__main__":
    unittest.main()
