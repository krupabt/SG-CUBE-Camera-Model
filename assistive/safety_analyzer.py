import cv2
import numpy as np
from typing import Dict, List, Optional
from .spatial_analyzer import SpatialAnalyzer

class SafetyAnalyzer:
    """
    Real-time Hazard Perception Layer.
    Detects potential physical obstacles, stairs/steps, doors, close walls, and close objects.
    IMPORTANT: Emits conservative warnings and NEVER guarantees absolute safety.
    """

    def __init__(self, spatial_analyzer: SpatialAnalyzer):
        self.spatial = spatial_analyzer
        self.last_hazard_warning: str = ""

    def analyze_hazards(self, frame: np.ndarray, face_results: Optional[List[Dict]] = None) -> Dict:
        """
        Analyzes camera frame for potential safety hazards.
        Returns dict with: 'hazard_detected', 'hazard_type', 'warning_text', 'urgency'
        """
        if frame is None or frame.size == 0:
            return {
                "hazard_detected": False,
                "hazard_type": None,
                "warning_text": "",
                "urgency": "none"
            }

        h_img, w_img = frame.shape[:2]
        self.spatial.update_frame_dimensions(w_img, h_img)

        # 1. Close Face / Person Detection
        if face_results:
            for face in face_results:
                x, y, w, h = face["bbox"]
                if h / float(h_img) > 0.55:  # Face occupies more than 55% height -> very close
                    return {
                        "hazard_detected": True,
                        "hazard_type": "close_person",
                        "warning_text": "Person very close in front of you.",
                        "urgency": "high"
                    }

        # 2. Horizontal Line Detection for Stairs / Steps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=w_img*0.4, maxLineGap=10)

        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                pts = line.ravel()
                if len(pts) >= 4:
                    x1, y1, x2, y2 = pts[:4]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < 15:  # Nearly horizontal line
                        horizontal_lines += 1

        if horizontal_lines >= 4:
            return {
                "hazard_detected": True,
                "hazard_type": "stairs_or_steps",
                "warning_text": "Stairs or steps detected ahead. Please proceed with caution.",
                "urgency": "high"
            }

        # 3. Large Blocking Obstacle in Center Path
        center_region = gray[int(h_img*0.3):int(h_img*0.8), int(w_img*0.3):int(w_img*0.7)]
        variance = np.var(center_region)
        # Uniform low variance close-up -> close wall or solid obstacle blocking view
        if variance < 80.0:
            return {
                "hazard_detected": True,
                "hazard_type": "close_wall",
                "warning_text": "Possible close wall or obstacle directly ahead.",
                "urgency": "medium"
            }

        return {
            "hazard_detected": False,
            "hazard_type": None,
            "warning_text": "",
            "urgency": "none"
        }
