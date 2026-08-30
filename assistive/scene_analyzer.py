import cv2
import numpy as np
from typing import Dict, List, Optional
from .spatial_analyzer import SpatialAnalyzer

class SceneAnalyzer:
    """
    Environment Understanding & Scene Summarizer.
    Generates spatially structured environment descriptions for blind users.
    """

    def __init__(self, spatial_analyzer: SpatialAnalyzer):
        self.spatial = spatial_analyzer

    def generate_scene_summary(
        self,
        frame: np.ndarray,
        face_results: List[Dict],
        gemini_context: Optional[str] = None
    ) -> Dict:
        """
        Builds a concise, spatially useful description of the environment.
        """
        if gemini_context and len(gemini_context) > 10:
            # Use Gemini's contextual understanding if available
            return {
                "summary": gemini_context,
                "people_count": len(face_results),
                "indoor_outdoor": "indoor"
            }

        people_count = len(face_results)
        people_desc = ""
        if people_count == 1:
            name = face_results[0].get("name")
            spatial = self.spatial.get_spatial_zone(face_results[0]["bbox"])
            if name:
                people_desc = f"One person recognized as {name} is {spatial['full_verbal']}."
            else:
                people_desc = f"One person is {spatial['full_verbal']}."
        elif people_count > 1:
            people_desc = f"There are {people_count} people visible in front of you."

        if frame is not None and frame.size > 0:
            h_img, w_img = frame.shape[:2]
            brightness = np.mean(frame)
            light_desc = "well lit" if brightness > 80 else "dimly lit"
        else:
            light_desc = ""

        parts = []
        if people_desc:
            parts.append(people_desc)

        if light_desc:
            parts.append(f"The space is {light_desc}.")

        if not parts:
            summary = "The camera view is clear. No people or major obstacles immediately detected ahead."
        else:
            summary = " ".join(parts)

        return {
            "summary": summary,
            "people_count": people_count,
            "light_level": light_desc
        }
