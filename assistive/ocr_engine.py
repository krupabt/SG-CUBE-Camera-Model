import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

class OCREngine:
    """
    Dedicated OCR Text Reading System with text-change debouncing.
    Extracts text regions, filters noise, and prevents repeating unchanged text.
    """

    def __init__(self, repeat_cooldown: float = 15.0):
        self.repeat_cooldown = repeat_cooldown
        self.last_read_text: str = ""
        self.last_read_time: float = 0.0

    def detect_text_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects bounding boxes of candidate text areas using MSER / adaptive thresholding contours.
        """
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_boxes = []

        h_img, w_img = frame.shape[:2]
        min_area = (w_img * h_img) * 0.005

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect = w / float(h)
            if area > min_area and aspect > 1.2:
                text_boxes.append((x, y, w, h))

        return text_boxes

    def extract_simple_text_tokens(self, frame: np.ndarray) -> str:
        """
        Basic morphological text-line estimator for fast local text presence check.
        Can be complemented by EasyOCR/Tesseract or Gemini Live Vision OCR.
        """
        boxes = self.detect_text_regions(frame)
        if not boxes:
            return ""

        return f"Visible text in {len(boxes)} regions"

    def process_ocr(self, frame: np.ndarray, external_text: Optional[str] = None) -> Dict:
        """
        Processes OCR reading for the frame.
        """
        now = time.time()
        text_content = external_text.strip() if external_text else ""

        if not text_content:
            boxes = self.detect_text_regions(frame)
            if not boxes:
                return {
                    "text": "",
                    "has_text": False,
                    "is_new": False,
                    "description": "No text detected."
                }
            text_content = self.extract_simple_text_tokens(frame)

        # Clean text
        clean_text = " ".join(text_content.split())

        # Check for change / repeat cooldown
        is_new = False
        if clean_text != self.last_read_text or (now - self.last_read_time) > self.repeat_cooldown:
            is_new = True
            self.last_read_text = clean_text
            self.last_read_time = now

        return {
            "text": clean_text,
            "has_text": bool(clean_text),
            "is_new": is_new,
            "description": f"Text read: {clean_text}" if clean_text else "No text found."
        }
