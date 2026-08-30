import unittest
import numpy as np
from assistive.ocr_engine import OCREngine

class TestOCREngine(unittest.TestCase):
    def setUp(self):
        self.ocr = OCREngine(repeat_cooldown=5.0)

    def test_text_debouncing(self):
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        res1 = self.ocr.process_ocr(dummy_frame, external_text="SPEED LIMIT 50")
        self.assertTrue(res1["is_new"])
        self.assertEqual(res1["text"], "SPEED LIMIT 50")

        # Second call with same text should not be new
        res2 = self.ocr.process_ocr(dummy_frame, external_text="SPEED LIMIT 50")
        self.assertFalse(res2["is_new"])

        # Change text
        res3 = self.ocr.process_ocr(dummy_frame, external_text="STOP SIGN")
        self.assertTrue(res3["is_new"])
        self.assertEqual(res3["text"], "STOP SIGN")

if __name__ == "__main__":
    unittest.main()
