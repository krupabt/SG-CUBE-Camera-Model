from typing import Tuple, Dict

class SpatialAnalyzer:
    """
    Translates pixel bounding box coordinates into intuitive spatial verbal cues.
    Zoning:
    - Horizontal: LEFT, CENTER, RIGHT
    - Vertical: TOP, MIDDLE, BOTTOM
    - Distance estimate: close, near, farther ahead based on bounding box height ratio.
    """

    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.w = frame_width
        self.h = frame_height

    def update_frame_dimensions(self, width: int, height: int):
        self.w = max(1, width)
        self.h = max(1, height)

    def get_spatial_zone(self, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Calculates spatial position dict for bbox (x, y, w, h).
        Returns dict with:
        'h_zone': 'left' | 'center' | 'right'
        'v_zone': 'top' | 'middle' | 'bottom'
        'verbal_location': e.g. "slightly to your left", "directly ahead", "on your right"
        'distance_verbal': e.g. "very close", "near", "farther ahead"
        """
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        rel_x = cx / float(self.w)
        rel_y = cy / float(self.h)
        rel_h = h / float(self.h)

        # Horizontal zoning
        if rel_x < 0.38:
            h_zone = "left"
            h_verbal = "to your left"
        elif rel_x > 0.62:
            h_zone = "right"
            h_verbal = "on your right"
        else:
            h_zone = "center"
            h_verbal = "directly ahead"

        if 0.33 <= rel_x < 0.38:
            h_verbal = "slightly to your left"
        elif 0.62 < rel_x <= 0.67:
            h_verbal = "slightly to your right"

        # Vertical zoning
        if rel_y < 0.35:
            v_zone = "top"
        elif rel_y > 0.65:
            v_zone = "bottom"
        else:
            v_zone = "middle"

        # Distance estimation based on height ratio in camera view
        if rel_h > 0.60:
            distance_verbal = "very close"
            approx_dist = "less than 1 meter"
        elif rel_h > 0.30:
            distance_verbal = "near"
            approx_dist = "approximately 1 to 2 meters away"
        else:
            distance_verbal = "farther ahead"
            approx_dist = "more than 2 meters away"

        full_verbal = f"{h_verbal}, {distance_verbal}"

        return {
            "h_zone": h_zone,
            "v_zone": v_zone,
            "h_verbal": h_verbal,
            "distance_verbal": distance_verbal,
            "approx_dist": approx_dist,
            "full_verbal": full_verbal,
            "rel_x": rel_x,
            "rel_y": rel_y,
            "rel_h": rel_h
        }
