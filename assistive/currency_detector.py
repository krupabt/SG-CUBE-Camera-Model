import cv2
import numpy as np
import re
from typing import Dict, Optional, Tuple

class CurrencyDetector:
    """
    Dedicated Indian Banknote (INR) Recognition System.
    Analyzes visual dimensions, dominant color profiles, aspect ratio,
    and numeral text cues.
    """

    # Indian Banknote Color Profiles (HSV ranges) & Aspect Ratios
    # ₹10: Chocolate Brown
    # ₹20: Greenish Yellow
    # ₹50: Fluorescent Blue
    # ₹100: Lavender / Purple
    # ₹200: Bright Yellow / Orange-Yellow
    # ₹500: Stone Grey
    # ₹2000: Magenta

    DENOMINATION_PROFILES = {
        "500": {
            "name": "500-rupee note",
            "val": 500,
            "hsv_lower": np.array([0, 0, 80]),     # Greyish / Stone Grey
            "hsv_upper": np.array([180, 45, 200]),
            "keywords": ["500", "FIVE HUNDRED", "RUPEES", "RESERVE BANK"]
        },
        "200": {
            "name": "200-rupee note",
            "val": 200,
            "hsv_lower": np.array([15, 120, 150]), # Bright Yellow/Orange
            "hsv_upper": np.array([35, 255, 255]),
            "keywords": ["200", "TWO HUNDRED"]
        },
        "100": {
            "name": "100-rupee note",
            "val": 100,
            "hsv_lower": np.array([125, 40, 100]), # Lavender / Purple
            "hsv_upper": np.array([155, 200, 255]),
            "keywords": ["100", "ONE HUNDRED"]
        },
        "50": {
            "name": "50-rupee note",
            "val": 50,
            "hsv_lower": np.array([85, 80, 120]),  # Fluorescent Blue / Cyan
            "hsv_upper": np.array([105, 255, 255]),
            "keywords": ["50", "FIFTY"]
        },
        "20": {
            "name": "20-rupee note",
            "val": 20,
            "hsv_lower": np.array([30, 80, 100]),  # Greenish Yellow
            "hsv_upper": np.array([45, 255, 255]),
            "keywords": ["20", "TWENTY"]
        },
        "10": {
            "name": "10-rupee note",
            "val": 10,
            "hsv_lower": np.array([8, 60, 60]),    # Chocolate Brown
            "hsv_upper": np.array([22, 180, 160]),
            "keywords": ["10", "TEN"]
        },
        "2000": {
            "name": "2000-rupee note",
            "val": 2000,
            "hsv_lower": np.array([160, 80, 120]), # Magenta
            "hsv_upper": np.array([178, 255, 255]),
            "keywords": ["2000", "TWO THOUSAND"]
        }
    }

    def __init__(self):
        pass

    def analyze_banknote(self, frame: np.ndarray, detected_text: Optional[str] = None) -> Dict:
        """
        Analyzes frame to identify Indian currency banknote denomination.
        Returns dict with keys: 'denom', 'name', 'confidence', 'description', 'certain'
        """
        if frame is None or frame.size == 0:
            return {
                "denom": None,
                "name": None,
                "confidence": "low",
                "description": "No note visible.",
                "certain": False
            }

        # 1. Text numeral search if text is provided
        text_matches = []
        if detected_text:
            text_upper = detected_text.upper()
            for key, prof in self.DENOMINATION_PROFILES.items():
                for kw in prof["keywords"]:
                    if re.search(r'\b' + re.escape(kw) + r'\b', text_upper):
                        text_matches.append((key, 0.9))

        # 2. Visual color histogram & contour region analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        total_pixels = frame.shape[0] * frame.shape[1]
        color_scores = {}

        for key, prof in self.DENOMINATION_PROFILES.items():
            mask = cv2.inRange(hsv, prof["hsv_lower"], prof["hsv_upper"])
            match_pixels = cv2.countNonZero(mask)
            ratio = match_pixels / float(total_pixels)
            color_scores[key] = ratio

        sorted_color = sorted(color_scores.items(), key=lambda x: x[1], reverse=True)
        top_color_key, top_color_ratio = sorted_color[0]

        # Combine text and color signals
        final_key = None
        confidence = "low"
        certain = False

        if text_matches:
            # Text signal is strong
            final_key = text_matches[0][0]
            if top_color_ratio > 0.08 and top_color_key == final_key:
                confidence = "high"
                certain = True
            else:
                confidence = "medium"
                certain = True
        elif top_color_ratio > 0.15:
            # Color signal is moderately strong
            final_key = top_color_key
            if top_color_ratio > 0.35:
                confidence = "medium"
                certain = True
            else:
                confidence = "low"
                certain = False
        else:
            final_key = None

        if final_key and final_key in self.DENOMINATION_PROFILES:
            prof = self.DENOMINATION_PROFILES[final_key]
            name = prof["name"]
            val = prof["val"]
            if certain and confidence == "high":
                desc = f"This is a {val}-rupee Indian banknote."
            elif certain:
                desc = f"This appears to be a {val}-rupee Indian banknote."
            else:
                desc = f"I think this is a {val}-rupee note, but I'm not completely sure. Please hold it steady closer to the camera."

            return {
                "denom": val,
                "name": name,
                "confidence": confidence,
                "description": desc,
                "certain": certain
            }

        return {
            "denom": None,
            "name": None,
            "confidence": "low",
            "description": "I cannot clearly identify the currency banknote. Please position the note clearly under good lighting.",
            "certain": False
        }
