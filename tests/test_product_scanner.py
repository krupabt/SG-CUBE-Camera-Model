import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assistive.product_scanner import ProductScanner

class TestProductScanner(unittest.TestCase):

    def setUp(self):
        self.scanner = ProductScanner()

    def test_medicine_detection(self):
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        ocr_text = "Paracetamol 500mg Tablets Take 1 daily EXP: 12/2026"
        res = self.scanner.scan_product_label(dummy_frame, ocr_text=ocr_text)

        self.assertTrue(res["is_medicine"])
        self.assertIsNotNone(res["expiry_date"])
        self.assertIn("12/2026", res["expiry_date"])
        self.assertIn("medicine", res["description"].lower())

    def test_perishable_food_detection(self):
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        ocr_text = "Whole Milk Organic EXP: 08/2027"
        res = self.scanner.scan_product_label(dummy_frame, ocr_text=ocr_text)

        self.assertFalse(res["is_medicine"])
        self.assertIsNotNone(res["expiry_date"])
        self.assertIn("08/2027", res["expiry_date"])

if __name__ == "__main__":
    unittest.main()
