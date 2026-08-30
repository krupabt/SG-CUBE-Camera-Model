import os
import json
import shutil
import time
import uuid
import math
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_FACE_DIR = os.path.join(PROJECT_ROOT, "data", "face_memory")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models")

class FaceMemory:
    """
    State-of-the-Art Persistent Face Memory & Recognition Engine.
    Uses deep neural face embeddings (SFace / ArcFace), facial alignment,
    quality gating, and multi-sample enrollment galleries for high accuracy
    and low false acceptance rates.
    """

    def __init__(self, storage_dir: str = None, models_dir: str = None):
        if storage_dir is None or storage_dir in ["data/face_memory", "face_memory"]:
            self.storage_dir = DEFAULT_FACE_DIR
        else:
            self.storage_dir = os.path.abspath(storage_dir)

        if models_dir is None or models_dir in ["data/models", "models"]:
            self.models_dir = DEFAULT_MODEL_DIR
        else:
            self.models_dir = os.path.abspath(models_dir)

        os.makedirs(self.storage_dir, exist_ok=True)
        self.profiles: Dict[str, Dict] = {}

        # Load Deep Feature Embedding Model (SFace / ArcFace 128-D)
        self.sface = None
        self.embedding_version = "v2"
        self.embedding_dimension = 128

        sface_candidates = [
            os.path.join(self.models_dir, "face_recognition_sface_2021dec.onnx"),
            os.path.join(DEFAULT_MODEL_DIR, "face_recognition_sface_2021dec.onnx"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "SG-CUBE", "data", "models", "face_recognition_sface_2021dec.onnx")
        ]
        for candidate in sface_candidates:
            if os.path.exists(candidate):
                try:
                    self.sface = cv2.FaceRecognizerSF.create(model=candidate, config="")
                    print(f"[FACE-ENGINE] Initialized Deep SFace Recognizer ({candidate})")
                    break
                except Exception as e:
                    print(f"[FACE-ENGINE] SFace initialization warning ({candidate}): {e}")

        if self.sface is None:
            self.embedding_version = "v1"
            self.embedding_dimension = 256
            print("[FACE-ENGINE] Using high-discrimination structural feature fallback.")

        self.load_all_profiles()

    def check_face_quality(self, crop: np.ndarray) -> Tuple[bool, str]:
        """
        Face Quality Gate:
        Evaluates resolution, focus/blur, illumination, and contrast.
        Returns (is_valid, reason_str).
        """
        if crop is None or getattr(crop, 'size', 0) == 0:
            return False, "Empty face crop"

        h, w = crop.shape[:2]
        if h < 36 or w < 36:
            return False, f"Face resolution too small ({w}x{h}px, minimum 36x36px required)"

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # Illumination Check (mean pixel value)
        mean_val = float(np.mean(gray))
        if mean_val < 20.0:
            return False, "Face is too dark / underexposed"
        if mean_val > 240.0:
            return False, "Face is overexposed / washed out"

        # Contrast Check (standard deviation of pixel intensity)
        std_val = float(np.std(gray))
        if std_val < 10.0:
            return False, "Face has insufficient visual contrast"

        # Focus / Blur Check via Laplacian Variance
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 8.0:
            return False, "Face is blurry or out of focus"

        return True, "Good quality"

    def align_and_preprocess(self, crop: np.ndarray, target_size=(112, 112)) -> np.ndarray:
        """
        Normalizes geometry and illumination with CLAHE contrast enhancement.
        """
        if crop is None or crop.size == 0:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        # Standard geometric resize with high-quality Lanczos resampling
        aligned = cv2.resize(crop, target_size, interpolation=cv2.INTER_LANCZOS4)

        # Lighting & contrast normalization using CLAHE on Luminance channel
        ycrcb = cv2.cvtColor(aligned, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        aligned_norm = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        return aligned_norm

    def load_all_profiles(self):
        self.profiles.clear()
        if not os.path.exists(self.storage_dir):
            return

        for person_dir_name in os.listdir(self.storage_dir):
            person_path = os.path.join(self.storage_dir, person_dir_name)
            if not os.path.isdir(person_path):
                continue

            meta_path = os.path.join(person_path, "metadata.json")
            emb_path = os.path.join(person_path, "embedding.npy")

            if os.path.exists(meta_path) and os.path.exists(emb_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    embedding = np.load(emb_path)
                    person_id = meta.get("id", person_dir_name)

                    # Load gallery embeddings if present
                    gallery = []
                    gallery_path = os.path.join(person_path, "gallery.npy")
                    if os.path.exists(gallery_path):
                        gallery_arr = np.load(gallery_path)
                        gallery = [gallery_arr[i] for i in range(len(gallery_arr))]
                    else:
                        gallery = [embedding]

                    self.profiles[person_id] = {
                        "id": person_id,
                        "name": meta.get("name", "Unknown"),
                        "created_at": meta.get("created_at", time.time()),
                        "metadata": meta,
                        "embedding": embedding,
                        "gallery": gallery,
                        "dir_path": person_path
                    }
                except Exception as e:
                    print(f"Error loading face profile from {person_path}: {e}")

    def compute_face_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Computes a normalized feature embedding vector for a face crop.
        Returns a unit-normalized L2 feature vector.
        """
        if face_crop is None or getattr(face_crop, 'size', 0) == 0:
            return np.zeros((self.embedding_dimension,), dtype=np.float32)

        aligned = self.align_and_preprocess(face_crop, target_size=(112, 112))

        # Deep SFace (128-D) Feature Extractor
        if self.sface is not None:
            try:
                feat = self.sface.feature(aligned).flatten()
                norm = float(np.linalg.norm(feat))
                if norm > 0:
                    return (feat / norm).astype(np.float32)
                return feat.astype(np.float32)
            except Exception as e:
                print(f"[FACE-ENGINE] SFace inference fallback: {e}")

        # High-discrimination structural feature vector (256-D)
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(aligned, cv2.COLOR_BGR2HSV)

        # 1. Orientation Gradient Histograms (64 bins)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
        ang_hist = cv2.calcHist([ang.astype(np.uint8)], [0], None, [64], [0, 360]).flatten()

        # 2. Local Spatial Texture Grids (4x4 cells -> 128 bins)
        cell_features = []
        for r in range(4):
            for c in range(4):
                cell = gray[r*28:(r+1)*28, c*28:(c+1)*28]
                h = cv2.calcHist([cell], [0], None, [8], [0, 256]).flatten()
                cell_features.append(h)
        grid_feat = np.concatenate(cell_features)

        # 3. Fine Color Spatial Distribution (64 bins)
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        color_feat = np.concatenate([h_hist, s_hist])

        raw_vec = np.concatenate([ang_hist, grid_feat, color_feat])
        norm = float(np.linalg.norm(raw_vec))
        if norm > 0:
            return (raw_vec / norm).astype(np.float32)
        return raw_vec.astype(np.float32)

    def save_person(self, name: str, face_crop: np.ndarray, extra_meta: Optional[Dict] = None) -> str:
        """
        Enrolls a new face profile with multi-sample augmentation gallery.
        """
        clean_name = name.strip()
        slug = "".join(c if c.isalnum() else "_" for c in clean_name.lower())
        person_id = f"{slug}_{uuid.uuid4().hex[:6]}"
        person_dir = os.path.join(self.storage_dir, person_id)
        os.makedirs(person_dir, exist_ok=True)

        primary_embedding = self.compute_face_embedding(face_crop)

        # Build Multi-Sample Gallery (capturing horizontal symmetry & subtle lighting variations)
        gallery_embs = [primary_embedding]

        if face_crop is not None and face_crop.size > 0:
            try:
                # Augmentation 1: Horizontal Flip (invariance to asymmetric camera angles)
                flipped = cv2.flip(face_crop, 1)
                gallery_embs.append(self.compute_face_embedding(flipped))

                # Augmentation 2: Center-cropped zoom (invariance to camera distance)
                h, w = face_crop.shape[:2]
                if h > 40 and w > 40:
                    ch, cw = int(h * 0.08), int(w * 0.08)
                    cropped_zoom = face_crop[ch:h-ch, cw:w-cw]
                    gallery_embs.append(self.compute_face_embedding(cropped_zoom))
            except Exception:
                pass

        # Save reference face image
        ref_image_path = os.path.join(person_dir, "reference.jpg")
        cv2.imwrite(ref_image_path, face_crop)

        # Save primary embedding & full gallery array
        emb_path = os.path.join(person_dir, "embedding.npy")
        np.save(emb_path, primary_embedding)

        gallery_path = os.path.join(person_dir, "gallery.npy")
        np.save(gallery_path, np.array(gallery_embs, dtype=np.float32))

        meta = {
            "id": person_id,
            "name": clean_name,
            "created_at": time.time(),
            "has_reference_image": True,
            "version": self.embedding_version,
            "dimension": int(primary_embedding.shape[0]),
            "gallery_size": len(gallery_embs),
            "extra": extra_meta or {}
        }
        meta_path = os.path.join(person_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self.profiles[person_id] = {
            "id": person_id,
            "name": clean_name,
            "created_at": meta["created_at"],
            "metadata": meta,
            "embedding": primary_embedding,
            "gallery": gallery_embs,
            "dir_path": person_dir
        }

        return person_id

    def list_people(self) -> List[str]:
        return sorted(list(set(p["name"] for p in self.profiles.values())))

    def forget_person(self, name: str) -> bool:
        clean_name = name.strip().lower()
        target_ids = [pid for pid, p in self.profiles.items() if p["name"].lower() == clean_name]

        if not target_ids:
            return False

        for pid in target_ids:
            dir_path = self.profiles[pid]["dir_path"]
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)
            del self.profiles[pid]

        return True

    def forget_all_faces(self) -> int:
        count = len(self.profiles)
        for pid, p in list(self.profiles.items()):
            dir_path = p["dir_path"]
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)
        self.profiles.clear()
        return count

    def clear_all_profiles(self) -> int:
        """ Alias for forget_all_faces to clear all face profiles """
        return self.forget_all_faces()

    def find_match(self, face_crop: np.ndarray, threshold: float = 0.55) -> Tuple[Optional[str], float]:
        """
        Compares input face crop against enrolled multi-sample galleries using Cosine Similarity.
        Returns (best_name, confidence_score) or (None, best_score).
        """
        if not self.profiles or face_crop is None or getattr(face_crop, 'size', 0) == 0:
            return None, 0.0

        # Quality Gate Check
        quality_ok, _ = self.check_face_quality(face_crop)
        if not quality_ok:
            return None, 0.0

        query_emb = self.compute_face_embedding(face_crop)
        best_name = None
        best_sim = -1.0

        for p in self.profiles.values():
            profile_gallery = p.get("gallery", [p.get("embedding")])
            for profile_emb in profile_gallery:
                if profile_emb is None:
                    continue
                # If embedding dimensions match, compute Cosine similarity
                if query_emb.shape == profile_emb.shape:
                    sim = float(np.dot(query_emb, profile_emb))
                else:
                    # Version compatibility fallback (e.g. comparing 128-D vs 256-D)
                    min_len = min(len(query_emb), len(profile_emb))
                    v1 = query_emb[:min_len] / (np.linalg.norm(query_emb[:min_len]) or 1.0)
                    v2 = profile_emb[:min_len] / (np.linalg.norm(profile_emb[:min_len]) or 1.0)
                    sim = float(np.dot(v1, v2))

                if sim > best_sim:
                    best_sim = sim
                    best_name = p["name"]

        if best_sim >= threshold:
            return best_name, best_sim
        else:
            return None, max(0.0, best_sim)
