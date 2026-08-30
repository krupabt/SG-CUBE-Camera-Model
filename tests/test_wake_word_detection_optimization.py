import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wake_word_matcher import WakeWordMatcher

class TestDedicatedWakeWordDetector(unittest.TestCase):

    def test_positive_phonetic_variants(self):
        positives = [
            "SG CUBE",
            "Hey SG CUBE",
            "SG",
            "S G",
            "Cube",
            "SG Cub",
            "SG Cue",
            "SGQ",
            "S G Q",
            "Ess Gee",
            "Ess Gee Cube",
            "Es Gee",
            "Es Gee Cube",
            "ESG",
            "ESG Cube",
            "Hey SG",
            "Hey S G",
            "Hey Ess Gee",
            "Hey Ess Gee Cube"
        ]
        for text in positives:
            is_match, conf, reason, norm, dbg = WakeWordMatcher.evaluate(text)
            self.assertTrue(is_match, f"Failed positive hotword variant: '{text}' (reason={reason})")
            self.assertGreaterEqual(conf, 0.70)

    def test_negative_non_wake_rejection(self):
        negatives = [
            "Rahul",
            "Sahana",
            "Alexth",
            "hello",
            "facebook",
            "execute",
            "play",
            "google",
            "youtube",
            "computer",
            "good morning",
            "how are you",
            "normal conversation"
        ]
        for text in negatives:
            is_match, conf, reason, norm, dbg = WakeWordMatcher.evaluate(text)
            self.assertFalse(is_match, f"Failed negative non-wake rejection (should REJECT): '{text}' (conf={conf}, reason={reason})")

    def test_single_trigger_debounce(self):
        last_wake_time = 0.0
        trigger_count = 0

        for offset in [0.0, 0.2, 0.4]:
            now = time.time() + offset
            if now - last_wake_time > 2.0:
                is_match, conf, reason, norm, dbg = WakeWordMatcher.evaluate("Hey SG CUBE")
                if is_match:
                    last_wake_time = now
                    trigger_count += 1

        self.assertEqual(trigger_count, 1, "Debounce failed: expected exactly 1 trigger per utterance")

if __name__ == "__main__":
    unittest.main()
