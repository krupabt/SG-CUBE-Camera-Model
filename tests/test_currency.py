import unittest
import numpy as np
from assistive.currency_detector import CurrencyDetector

class TestCurrencyDetector(unittest.TestCase):
    def setUp(self):
        self.detector = CurrencyDetector()

    def test_text_based_currency(self):
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        res = self.detector.analyze_banknote(dummy_frame, detected_text="RESERVE BANK OF INDIA 500 RUPEES")
        self.assertEqual(res["denom"], 500)
        self.assertTrue(res["certain"])
        self.assertEqual(res["confidence"], "medium")

    def test_unknown_currency(self):
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        res = self.detector.analyze_banknote(dummy_frame, detected_text=None)
        self.assertIsNone(res["denom"])
        self.assertFalse(res["certain"])

if __name__ == "__main__":
    unittest.main()
