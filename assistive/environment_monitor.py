import time
import numpy as np
from typing import Dict, List, Optional

class EnvironmentMonitor:
    """
    Manages On-Demand vs Continuous Assistive Monitoring modes.
    Evaluates scene changes and triggers spoken announcements only when material changes occur.
    """

    def __init__(self, announcement_cooldown: float = 10.0):
        self.mode = "on_demand"  # "on_demand" or "continuous"
        self.announcement_cooldown = announcement_cooldown
        self.last_announcement_time: float = 0.0
        self.last_scene_fingerprint: Optional[float] = None
        self.last_people_count: int = 0
        self.last_hazard: Optional[str] = None

    def set_mode(self, mode: str):
        if mode in ("on_demand", "continuous"):
            self.mode = mode

    def is_continuous(self) -> bool:
        return self.mode == "continuous"

    def calculate_frame_fingerprint(self, frame: np.ndarray) -> float:
        if frame is None or frame.size == 0:
            return 0.0
        small = np.mean(frame, axis=(0, 1))
        return float(np.sum(small))

    def evaluate_continuous_events(
        self,
        frame: np.ndarray,
        face_results: List[Dict],
        safety_result: Dict,
        ocr_result: Dict,
        target_object_result: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Evaluates whether a continuous monitoring announcement should be fired.
        Returns notification dict or None.
        """
        if self.mode != "continuous":
            return None

        now = time.time()
        if (now - self.last_announcement_time) < self.announcement_cooldown:
            return None

        # 1. Check Safety Alert (highest priority continuous event)
        if safety_result.get("hazard_detected"):
            hazard_type = safety_result.get("hazard_type")
            if hazard_type != self.last_hazard:
                self.last_hazard = hazard_type
                self.last_announcement_time = now
                return {
                    "type": "safety",
                    "text": safety_result.get("warning_text"),
                    "priority": 1
                }
        else:
            self.last_hazard = None

        # 2. Check Greetings for Known Persons
        for face in face_results:
            if face.get("should_greet") and face.get("name"):
                self.last_announcement_time = now
                return {
                    "type": "greeting",
                    "text": f"{face['name']} is in front of you.",
                    "priority": 3
                }

        # 3. Check People Count Changes
        current_people = len(face_results)
        if current_people > 0 and current_people != self.last_people_count:
            self.last_people_count = current_people
            self.last_announcement_time = now
            if current_people == 1:
                msg = "A person has stepped into your camera view."
            else:
                msg = f"{current_people} people are now visible."
            return {
                "type": "people_change",
                "text": msg,
                "priority": 4
            }
        self.last_people_count = current_people

        # 4. Check Target Object Discovery
        if target_object_result and target_object_result.get("found"):
            self.last_announcement_time = now
            return {
                "type": "object_found",
                "text": target_object_result.get("response_text"),
                "priority": 4
            }

        return None
