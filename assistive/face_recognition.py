import time
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from .face_memory import FaceMemory

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models")

class FaceRecognizer:
    """
    High-Performance, Real-Time Face Recognition Pipeline.
    Combines Deep YuNet CNN detector + SFace ArcFace embedding,
    fast spatial-temporal tracking cache, and face quality gating.
    """

    def __init__(self, face_memory: FaceMemory, threshold: float = 0.55, greeting_cooldown: float = 30.0, models_dir: str = None):
        self.face_memory = face_memory
        self.threshold = threshold
        self.greeting_cooldown = greeting_cooldown
        self.greetings_enabled = True

        if models_dir is None or models_dir in ["data/models", "models"]:
            self.models_dir = DEFAULT_MODEL_DIR
        else:
            self.models_dir = os.path.abspath(models_dir)

        # Track recent greetings per person name -> timestamp
        self.greeting_history: Dict[str, float] = {}

        # Fast Spatial-Temporal Tracking Cache (bypasses heavy inference when face is stationary)
        self._last_recognition_results: List[Dict] = []
        self._last_recognition_time: float = 0.0
        self._recognition_cache_ttl: float = 0.15  # 150ms cache window

        # 1. Initialize SOTA Deep YuNet Face Detector
        self.yunet = None
        yunet_candidates = [
            os.path.join(self.models_dir, "face_detection_yunet_2023mar.onnx"),
            os.path.join(DEFAULT_MODEL_DIR, "face_detection_yunet_2023mar.onnx"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "SG-CUBE", "data", "models", "face_detection_yunet_2023mar.onnx")
        ]
        for candidate in yunet_candidates:
            if os.path.exists(candidate):
                try:
                    self.yunet = cv2.FaceDetectorYN.create(
                        model=candidate,
                        config="",
                        input_size=(320, 320),
                        score_threshold=0.55,
                        nms_threshold=0.3,
                        top_k=5000
                    )
                    print(f"[FACE-DETECTOR] Initialized Deep YuNet Face Detector ({candidate})")
                    break
                except Exception as e:
                    print(f"[FACE-DETECTOR] YuNet initialization warning: {e}")

        # 2. Haar Cascade fallback
        self.face_cascade = None
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                self.face_cascade = None

    def set_greetings_enabled(self, enabled: bool):
        self.greetings_enabled = enabled

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces in frame and returns list of bounding boxes (x, y, w, h).
        Uses Deep YuNet CNN when available, with Cascade fallback.
        """
        if frame is None or getattr(frame, 'size', 0) == 0:
            return []

        h_img, w_img = frame.shape[:2]

        # 1. Try Deep YuNet Detector
        if self.yunet is not None:
            try:
                self.yunet.setInputSize((w_img, h_img))
                _, detections = self.yunet.detect(frame)
                if detections is not None and len(detections) > 0:
                    face_boxes = []
                    for det in detections:
                        x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                        x = max(0, min(x, w_img - 1))
                        y = max(0, min(y, h_img - 1))
                        w = max(1, min(w, w_img - x))
                        h = max(1, min(h, h_img - y))
                        if w >= 24 and h >= 24:
                            face_boxes.append((x, y, w, h))
                    if face_boxes:
                        face_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
                        return face_boxes
            except Exception:
                pass

        # 2. Haar Cascade fallback
        if self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.08,
                    minNeighbors=4,
                    minSize=(30, 30)
                )
                if len(faces) > 0:
                    face_boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
                    face_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
                    return face_boxes
            except Exception:
                pass

        # 3. Robust skin-tone & contour fallback face detector
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (w_img * h_img) * 0.015

        face_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = h / float(w)
                if 0.7 <= aspect <= 2.2:
                    face_boxes.append((int(x), int(y), int(w), int(h)))

        face_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return face_boxes

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        High-Performance Face Processing with Spatial-Temporal Tracking Cache.
        Detects, quality-gates, and recognizes all faces present in frame.
        """
        results = []
        if frame is None or getattr(frame, 'size', 0) == 0:
            return results

        now = time.time()
        face_boxes = self.detect_faces(frame)

        # Spatial-Temporal Cache Hit Check
        if (now - self._last_recognition_time) < self._recognition_cache_ttl and len(face_boxes) == len(self._last_recognition_results):
            can_reuse = True
            for i, (x, y, w, h) in enumerate(face_boxes):
                cached = self._last_recognition_results[i]
                cx, cy, cw, ch = cached["bbox"]
                if abs(x - cx) > 25 or abs(y - cy) > 25:
                    can_reuse = False
                    break
            if can_reuse:
                return self._last_recognition_results

        for (x, y, w, h) in face_boxes:
            h_pad = int(h * 0.15)
            w_pad = int(w * 0.15)
            y1 = max(0, y - h_pad)
            y2 = min(frame.shape[0], y + h + h_pad)
            x1 = max(0, x - w_pad)
            x2 = min(frame.shape[1], x + w + w_pad)

            face_crop = frame[y1:y2, x1:x2]
            quality_ok, quality_reason = self.face_memory.check_face_quality(face_crop)

            if not quality_ok:
                results.append({
                    "bbox": (x, y, w, h),
                    "name": None,
                    "confidence": 0.0,
                    "crop": face_crop,
                    "quality_ok": False,
                    "quality_reason": quality_reason,
                    "should_greet": False
                })
                continue

            name, confidence = self.face_memory.find_match(face_crop, threshold=self.threshold)

            should_greet = False
            if name and self.greetings_enabled:
                last_greet = self.greeting_history.get(name, 0.0)
                if (now - last_greet) > self.greeting_cooldown:
                    should_greet = True
                    self.greeting_history[name] = now

            results.append({
                "bbox": (x, y, w, h),
                "name": name,
                "confidence": confidence,
                "crop": face_crop,
                "quality_ok": True,
                "quality_reason": "Good quality",
                "should_greet": should_greet
            })

        self._last_recognition_results = results
        self._last_recognition_time = now
        return results

    def get_primary_face_crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        boxes = self.detect_faces(frame)
        if not boxes:
            return None

        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = boxes[0]

        h_pad = int(h * 0.15)
        w_pad = int(w * 0.15)
        y1 = max(0, y - h_pad)
        y2 = min(frame.shape[0], y + h + h_pad)
        x1 = max(0, x - w_pad)
        x2 = min(frame.shape[1], x + w + w_pad)

        return frame[y1:y2, x1:x2]

    def enroll_active_face(self, frame: np.ndarray, name: str) -> Dict:
        if frame is None or getattr(frame, 'size', 0) == 0:
            return {"success": False, "message": "No visual frame available to enroll face."}

        crop = self.get_primary_face_crop(frame)
        if crop is None or crop.size == 0:
            return {
                "success": False,
                "message": f"I couldn't detect a face to save. Please look directly into the camera so I can remember {name}."
            }

        quality_ok, quality_reason = self.face_memory.check_face_quality(crop)
        if not quality_ok:
            return {
                "success": False,
                "message": f"Face quality too low for reliable recognition ({quality_reason}). Please look directly into the camera with good lighting so I can remember {name}."
            }

        person_id = self.face_memory.save_person(name=name, face_crop=crop)
        # Invalidate recognition cache on new enrollment
        self._last_recognition_time = 0.0
        return {
            "success": True,
            "person_id": person_id,
            "name": name,
            "message": f"Got it. I have remembered this face as {name}."
        }

    def detect_and_recognize_faces(self, frame: np.ndarray) -> List[Dict]:
        return self.process_frame(frame)
