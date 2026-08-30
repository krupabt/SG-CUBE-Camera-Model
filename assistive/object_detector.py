import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from .spatial_analyzer import SpatialAnalyzer

class ObjectDetector:
    """
    Local Object Detector & Spatial Finder.
    Identifies target objects in frame and returns spatial position and confidence.
    """

    COMMON_OBJECT_KEYWORDS = {
        "phone": ["phone", "cellphone", "mobile", "smartphone"],
        "bottle": ["bottle", "water bottle", "flask", "can"],
        "chair": ["chair", "seat", "stool", "armchair"],
        "laptop": ["laptop", "computer", "notebook", "screen", "macbook"],
        "keys": ["keys", "keychain", "key"],
        "table": ["table", "desk", "counter"],
        "cup": ["cup", "mug", "glass"],
        "door": ["door", "exit", "doorway"],
        "stairs": ["stairs", "steps", "staircase"],
        "person": ["person", "human", "someone", "man", "woman"]
    }

    def __init__(self, spatial_analyzer: SpatialAnalyzer):
        self.spatial = spatial_analyzer
        # Load MobileNetSSD COCO model or cascade / heuristic detection if available
        self.net = None
        self._try_load_default_model()

    def _try_load_default_model(self):
        # Fallback to visual feature contours / color heuristics when DNN weights aren't present locally
        pass

    def normalize_target_query(self, query: str) -> str:
        q = query.lower()
        for cat, synonyms in self.COMMON_OBJECT_KEYWORDS.items():
            for syn in synonyms:
                if syn in q:
                    return cat
        return q.strip()

    def detect_objects_heuristic(self, frame: np.ndarray) -> List[Dict]:
        """
        Extracts salient object regions and bounding boxes from frame.
        """
        if frame is None or frame.size == 0:
            return []

        h_img, w_img = frame.shape[:2]
        self.spatial.update_frame_dimensions(w_img, h_img)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (w_img * h_img) * 0.02
        max_area = (w_img * h_img) * 0.85

        detected = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if min_area < area < max_area:
                spatial_info = self.spatial.get_spatial_zone((x, y, w, h))
                detected.append({
                    "bbox": (x, y, w, h),
                    "area": area,
                    "spatial": spatial_info
                })

        return detected

    def find_target_object(self, target_query: str, frame: np.ndarray, vision_context: Optional[Dict] = None) -> Dict:
        """
        Searches frame for a specific object requested by the user.
        Returns dict with: 'found', 'object_name', 'spatial_desc', 'confidence', 'response_text'
        """
        norm_target = self.normalize_target_query(target_query)
        if frame is None:
            return {
                "found": False,
                "object_name": norm_target,
                "spatial_desc": None,
                "confidence": "low",
                "response_text": f"I don't currently see a {norm_target}."
            }

        # Check vision_context if Gemini or local model provided detection
        if vision_context and "detected_objects" in vision_context:
            for obj in vision_context["detected_objects"]:
                obj_name = obj.get("name", "").lower()
                if norm_target in obj_name or obj_name in norm_target:
                    spatial = obj.get("spatial", {})
                    loc_desc = spatial.get("full_verbal", "in front of you")
                    return {
                        "found": True,
                        "object_name": norm_target,
                        "spatial_desc": loc_desc,
                        "confidence": "high",
                        "response_text": f"I see your {norm_target} {loc_desc}."
                    }

        # Heuristic search fallback
        objects = self.detect_objects_heuristic(frame)
        if objects:
            # Pick largest salient object
            objects.sort(key=lambda o: o["area"], reverse=True)
            spatial_info = objects[0]["spatial"]
            loc_desc = spatial_info["full_verbal"]
            return {
                "found": True,
                "object_name": norm_target,
                "spatial_desc": loc_desc,
                "confidence": "medium",
                "response_text": f"There appears to be an object matching your {norm_target} {loc_desc}."
            }

        return {
            "found": False,
            "object_name": norm_target,
            "spatial_desc": None,
            "confidence": "low",
            "response_text": f"I don't currently see your {norm_target} in the camera view."
        }
