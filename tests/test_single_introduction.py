import unittest
from assistive.command_router import CommandRouter, OFFICIAL_INTRODUCTION
from assistive.vision_engine import VisionEngine

class TestSingleIntroduction(unittest.TestCase):
    def setUp(self):
        self.router = CommandRouter()
        self.engine = VisionEngine(data_dir="data")

    def test_single_introduction_no_preliminary_text(self):
        for i in range(5):
            resp = self.engine.process_user_speech_query("introduce yourself")
            self.assertEqual(resp, OFFICIAL_INTRODUCTION, f"Failed on run {i+1}")
            self.assertTrue(resp.startswith("Hi! I’m SG CUBE"))

    def test_who_are_you_exact_match(self):
        for i in range(5):
            resp = self.engine.process_user_speech_query("who are you")
            self.assertEqual(resp, OFFICIAL_INTRODUCTION, f"Failed on run {i+1}")

if __name__ == "__main__":
    unittest.main()