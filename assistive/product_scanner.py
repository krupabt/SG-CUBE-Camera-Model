import cv2
import re
import numpy as np
from typing import Dict, Optional, List

class ProductScanner:
    """
    Product, Barcode, QR Code, and Packaging Scanner for Assistive Vision.
    Detects QR codes, 1D barcodes, medicine label warnings, expiration dates,
    and package text to assist visually impaired users.
    """

    EXPIRY_PATTERNS = [
        r"(?:EXP|EXPIRY|USE BY|BEST BEFORE|BB)[:\s]*([0-9]{1,2}[/\.-][0-9]{2,4})",
        r"(?:EXP|EXPIRY|USE BY|BEST BEFORE|BB)[:\s]*([A-Z]{3,9}\s+[0-9]{2,4})",
        r"([0-9]{2}[/\.-][0-9]{4})"
    ]

    MEDICINE_KEYWORDS = [
        "mg", "ml", "tablets", "capsules", "syrup", "dosage",
        "paracetamol", "aspirin", "ibuprofen", "amoxicillin", "vitamin",
        "take", "daily", "prescription", "rx"
    ]

    def __init__(self):
        self.qr_detector = cv2.QRCodeDetector()

    def scan_qr_code(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detects and decodes QR codes present in the camera frame.
        """
        if frame is None or frame.size == 0:
            return None

        try:
            val, points, _ = self.qr_detector.detectAndDecode(frame)
            if val:
                return {
                    "type": "QR_CODE",
                    "data": val,
                    "description": f"QR Code detected containing: {val}"
                }
        except Exception:
            pass
        return None

    def scan_product_label(self, frame: np.ndarray, ocr_text: str = "") -> Dict:
        """
        Analyzes frame and OCR text for product details, barcodes, expiration dates, and medicine labels.
        """
        qr_result = self.scan_qr_code(frame)
        if qr_result:
            return qr_result

        # Search for Expiry Date in OCR text
        expiry_match = None
        for pattern in self.EXPIRY_PATTERNS:
            match = re.search(pattern, ocr_text, re.IGNORECASE)
            if match:
                expiry_match = match.group(0)
                break

        # Check for Medicine Packaging Indications
        ocr_lower = ocr_text.lower()
        is_medicine = any(kw in ocr_lower for kw in self.MEDICINE_KEYWORDS)

        if is_medicine:
            product_type = "Medicine / Pharmaceutical Item"
        elif expiry_match:
            product_type = "Perishable Food / Consumer Good"
        elif len(ocr_text) > 5:
            product_type = "Packaged Goods Label"
        else:
            product_type = "Unknown Item"

        # Build verbal response description
        desc_parts = []
        if is_medicine:
            desc_parts.append("This appears to be a medicine container.")
        elif product_type != "Unknown Item":
            desc_parts.append(f"Detected {product_type}.")

        if ocr_text.strip():
            desc_parts.append(f"Label text reads: '{ocr_text.strip()[:150]}'.")

        if expiry_match:
            desc_parts.append(f"Expiration notice: {expiry_match}.")
        elif not desc_parts:
            desc_parts.append("Hold the item label or QR code closer to the camera to scan details.")

        return {
            "type": "PRODUCT_LABEL",
            "product_type": product_type,
            "expiry_date": expiry_match,
            "is_medicine": is_medicine,
            "text": ocr_text,
            "description": " ".join(desc_parts)
        }
