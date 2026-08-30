import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assistive.memory_store import MemoryStore, DEFAULT_PREFERENCES

class TestVoiceConfiguration(unittest.TestCase):
    def test_default_male_voice_preference(self):
        self.assertEqual(DEFAULT_PREFERENCES.get("assistant_voice"), "Puck")

    def test_memory_store_voice_setting(self):
        store = MemoryStore(base_dir="data/test_voice_tmp")
        voice = store.get_setting("assistant_voice", "Puck")
        self.assertEqual(voice, "Puck")
        # Cleanup
        import shutil
        if os.path.exists("data/test_voice_tmp"):
            shutil.rmtree("data/test_voice_tmp", ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
