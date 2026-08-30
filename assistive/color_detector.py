import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

class ColorDetector:
    """
    Color Reader and Ambient Light Level Detector for Assistive Vision.
    Translates pixel region colors into human-friendly names (clothing/objects)
    and evaluates room illumination (lights on/off, dim, bright).
    """

    COLOR_MAP = [
        ("Black", (0, 0, 0), (180, 255, 45)),
        ("White", (0, 0, 200), (180, 30, 255)),
        ("Grey", (0, 0, 46), (180, 40, 199)),
        ("Red", (0, 100, 100), (10, 255, 255)),
        ("Red", (170, 100, 100), (180, 255, 255)),
        ("Orange", (11, 100, 100), (25, 255, 255)),
        ("Yellow", (26, 100, 100), (35, 255, 255)),
        ("Green", (36, 60, 60), (85, 255, 255)),
        ("Cyan / Light Blue", (86, 60, 60), (100, 255, 255)),
        ("Blue / Navy", (101, 60, 60), (130, 255, 255)),
        ("Purple / Violet", (131, 60, 60), (150, 255, 255)),
        ("Pink", (151, 60, 60), (169, 255, 255)),
        ("Brown / Beige", (10, 40, 50), (20, 180, 180)),
    ]

    def __init__(self):
        pass

    def check_ambient_light(self, frame: np.ndarray) -> Dict:
        """
        Calculates frame luminance and returns light intensity status.
        """
        if frame is None or frame.size == 0:
            return {"light_level": "UNKNOWN", "mean_luminance": 0.0, "description": "Unable to check lighting."}

        # Convert to LAB color space to extract L (Luminance) channel
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        mean_l = float(np.mean(l_channel))

        if mean_l < 30.0:
            level = "DARK"
            desc = "The room is very dark. The lights appear to be off."
        elif mean_l < 80.0:
            level = "DIM"
            desc = "The space is dimly lit."
        elif mean_l < 180.0:
            level = "NORMAL"
            desc = "The lighting conditions are normal and clear."
        else:
            level = "BRIGHT"
            desc = "The room is brightly lit or under direct sunlight."

        return {
            "light_level": level,
            "mean_luminance": round(mean_l, 2),
            "description": desc
        }

    def detect_dominant_color(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """
        Detects the dominant color name in the specified bounding box (x, y, w, h) or center ROI of frame.
        """
        if frame is None or frame.size == 0:
            return {"color_name": "Unknown", "confidence": 0.0, "description": "No visual frame available."}

        h, w = frame.shape[:2]
        if bbox is not None:
            bx, by, bw, bh = bbox
            # Clamp bounding box inside image bounds
            x1, y1 = max(0, bx), max(0, by)
            x2, y2 = min(w, bx + bw), min(h, by + bh)
            roi = frame[y1:y2, x1:x2]
        else:
            # Default to center region of interest (30% to 70% of dimensions)
            x1, y1 = int(w * 0.3), int(h * 0.3)
            x2, y2 = int(w * 0.7), int(h * 0.7)
            roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            roi = frame

        # Gaussian blur to remove high-frequency noise
        blurred = cv2.GaussianBlur(roi, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Count pixel matches across defined color ranges
        total_pixels = hsv.shape[0] * hsv.shape[1]
        color_counts: Dict[str, int] = {}

        for color_name, lower, upper in self.COLOR_MAP:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            count = int(cv2.countNonZero(mask))
            color_counts[color_name] = color_counts.get(color_name, 0) + count

        best_color = max(color_counts, key=color_counts.get) if color_counts else "Unknown"
        max_count = color_counts.get(best_color, 0)
        confidence = round(max_count / max(1, total_pixels), 2)

        if confidence < 0.05:
            best_color = "Neutral / Multi-color"

        return {
            "color_name": best_color,
            "confidence": confidence,
            "description": f"The dominant color appears to be {best_color.lower()}."
        }
