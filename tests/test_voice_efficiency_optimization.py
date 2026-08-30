import os
import sys
import time
import unittest
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visionclaw_gui import SGCubeApp

class TestVoiceEfficiencyOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.app = SGCubeApp(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.app.on_close()

    def test_single_authoritative_voice_config(self):
        voice_name = cls_voice = self.app.engine.store.get_setting("assistant_voice", "Puck")
        self.assertEqual(voice_name, "Puck")

    def test_instant_barge_in_queue_purge(self):
        initial_id = self.app.current_response_id
        # Queue 3 dummy audio chunks tagged with initial_id
        self.app.playback_queue.put((initial_id, b"dummy_pcm_audio_1"))
        self.app.playback_queue.put((initial_id, b"dummy_pcm_audio_2"))
        self.app.playback_queue.put((initial_id, b"dummy_pcm_audio_3"))

        self.assertFalse(self.app.playback_queue.empty())

        # Trigger barge-in queue clearing
        self.app._clear_playback_queue()

        # Confirm response_id incremented and playback queue is purged instantly
        self.assertGreater(self.app.current_response_id, initial_id)
        self.assertTrue(self.app.playback_queue.empty())

    def test_continuous_conversation_state(self):
        self.app.set_state("LISTENING")
        self.assertEqual(self.app.current_state, "LISTENING")

        self.app.set_state("USER_SPEAKING")
        self.assertEqual(self.app.current_state, "USER_SPEAKING")

        self.app.set_state("AI_SPEAKING")
        self.assertEqual(self.app.current_state, "AI_SPEAKING")

        self.app.set_state("LISTENING")
        self.assertEqual(self.app.current_state, "LISTENING")

if __name__ == "__main__":
    unittest.main()
