import os
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from .memory_store import MemoryStore
from .face_memory import FaceMemory
from .face_recognition import FaceRecognizer
from .currency_detector import CurrencyDetector
from .ocr_engine import OCREngine
from .spatial_analyzer import SpatialAnalyzer
from .object_detector import ObjectDetector
from .safety_analyzer import SafetyAnalyzer
from .scene_analyzer import SceneAnalyzer
from .environment_monitor import EnvironmentMonitor
from .command_router import CommandRouter, OFFICIAL_INTRODUCTION
from .response_manager import ResponseManager
from .memory_manager import MemoryManager
from .conversation_history import ConversationHistory
from .api_key_manager import APIKeyManager
from .color_detector import ColorDetector
from .product_scanner import ProductScanner
from .meta_glass import MetaGlassBridge

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

class VisionEngine:
    """
    Unified Perception Engine for VisionClaw Assistive Camera AI.
    Coordinates local face memory, persistent SQLite long-term memory, conversation history, API key manager,
    currency detection, OCR, object finding, spatial reasoning, safety hazard monitoring,
    color detection, product scanning, Meta Glass bridge, intent routing, and response queuing.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None or data_dir == "data":
            self.data_dir = DEFAULT_DATA_DIR
        else:
            self.data_dir = os.path.abspath(data_dir)

        self.store = MemoryStore(base_dir=self.data_dir)
        self.face_memory = FaceMemory(storage_dir=self.store.face_dir)
        self.memory = MemoryManager(db_dir=os.path.join(self.data_dir, "memory"))
        self.history = ConversationHistory(db_dir=os.path.join(self.data_dir, "history"))
        self.key_manager = APIKeyManager(pref_dir=self.store.pref_dir)
        self.meta_glass = MetaGlassBridge()

        threshold = self.store.get_setting("recognition_threshold", 0.55)
        cooldown = self.store.get_setting("greeting_cooldown_seconds", 30.0)
        self.face_recognizer = FaceRecognizer(
            face_memory=self.face_memory,
            threshold=threshold,
            greeting_cooldown=cooldown
        )
        self.face_recognizer.set_greetings_enabled(self.store.get_setting("greeting_enabled", True))

        self.currency_detector = CurrencyDetector()
        self.ocr_engine = OCREngine()
        self.color_detector = ColorDetector()
        self.product_scanner = ProductScanner()

        self.spatial = SpatialAnalyzer()
        self.object_detector = ObjectDetector(spatial_analyzer=self.spatial)
        self.safety = SafetyAnalyzer(spatial_analyzer=self.spatial)
        self.scene = SceneAnalyzer(spatial_analyzer=self.spatial)

        ann_cooldown = self.store.get_setting("announcement_cooldown_seconds", 10.0)
        self.monitor = EnvironmentMonitor(announcement_cooldown=ann_cooldown)
        if self.store.get_setting("environment_monitor_enabled", False):
            self.monitor.set_mode("continuous")

        self.router = CommandRouter()
        self.response_manager = ResponseManager(announcement_cooldown=ann_cooldown)

        # Per-frame perception state
        self.current_frame: Optional[np.ndarray] = None
        self.last_faces: List[Dict] = []
        self.last_safety: Dict = {}
        self.last_environment: Dict = {
            "light_level": "NORMAL",
            "light_desc": "Normal lighting",
            "scene_summary": "Clear space",
            "people_count": 0,
            "face_names": []
        }
        self.last_objects: List[Dict] = []
        self.active_mode: str = "ASSISTIVE"

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Per-frame perception processing. Runs lightweight face and safety checks.
        Returns per-frame state dict.
        """
        if frame is None or frame.size == 0:
            return {
                "faces": [],
                "safety": {},
                "environment": self.last_environment,
                "objects": self.last_objects,
                "mode": self.active_mode
            }

        self.current_frame = frame
        h_img, w_img = frame.shape[:2]
        self.spatial.update_frame_dimensions(w_img, h_img)

        # 1. Face Detection & Recognition
        faces = self.face_recognizer.process_frame(frame)
        self.last_faces = faces

        # 2. Safety Hazard Detection
        if self.store.get_setting("safety_alerts_enabled", True):
            safety_res = self.safety.analyze_hazards(frame, face_results=faces)
            self.last_safety = safety_res
            if safety_res.get("hazard_detected"):
                # Queue immediate safety response
                self.response_manager.add_response(safety_res["warning_text"], priority=1)
        else:
            safety_res = {}

        # 3. Ambient Environment & Scene Evaluation
        light_res = self.color_detector.check_ambient_light(frame)
        scene_res = self.scene.generate_scene_summary(frame, faces)
        self.last_environment = {
            "light_level": light_res.get("light_level", "NORMAL"),
            "light_desc": light_res.get("description", "Normal lighting"),
            "scene_summary": scene_res.get("summary", "Clear space"),
            "people_count": len(faces),
            "face_names": [f.get("name") or "Unknown" for f in faces]
        }

        # 4. Salient Objects & Spatial Detection
        objects_res = self.object_detector.detect_objects_heuristic(frame)
        self.last_objects = objects_res

        # 5. Continuous Assistive Monitor Evaluation
        if self.monitor.is_continuous():
            cont_event = self.monitor.evaluate_continuous_events(
                frame=frame,
                face_results=faces,
                safety_result=safety_res,
                ocr_result={}
            )
            if cont_event:
                self.response_manager.add_response(cont_event["text"], priority=cont_event.get("priority", 4))

        return {
            "faces": faces,
            "safety": safety_res,
            "environment": self.last_environment,
            "objects": objects_res,
            "mode": self.active_mode
        }

    def process_user_speech_query(self, user_transcript: str, session_id: Optional[str] = None) -> Optional[str]:
        """
        Routes user speech transcript to local perception & memory intent handlers.
        Returns immediate spoken response text if handled locally, or None to delegate to Gemini Live.
        """
        route = self.router.route_intent(user_transcript)
        intent = route["intent"]

        # --- PERSISTENT LONG-TERM MEMORY INTENTS ---
        if intent == "MEMORY_SAVE":
            raw_cmd = user_transcript
            norm_cmd = user_transcript.strip().lower()
            fact_str = route["params"].get("fact", "")
            key = route["params"].get("key", "fact")

            print(f"[SAVE] RAW COMMAND: '{raw_cmd}'")
            print(f"[SAVE] NORMALIZED COMMAND: '{norm_cmd}'")
            print(f"[SAVE] INTENT: 'MEMORY_SAVE'")
            print(f"[SAVE] KEY: '{key}'")
            print(f"[SAVE] FACT: '{fact_str}'")
            print(f"[SAVE] command received: '{raw_cmd}'")
            print(f"[SAVE] intent detected: 'MEMORY_SAVE'")
            print(f"[SAVE] memory handler started: 'MEMORY_SAVE'")

            # Contextual resolution for "Save this", "Save this information", "Remember this"
            if not fact_str or key == "contextual":
                last_msg = self.history.get_last_meaningful_message(session_id=session_id)
                if last_msg:
                    key_c, fact_c = self.router.extract_memory_key_and_fact(last_msg)
                    if fact_c:
                        key, fact_str = key_c, fact_c
                    else:
                        key = "contextual_fact"
                        fact_str = last_msg if last_msg.endswith(".") else last_msg + "."
                elif self.current_frame is not None:
                    ocr_res = self.ocr_engine.process_ocr(self.current_frame)
                    if ocr_res.get("has_text") and ocr_res.get("text"):
                        key = "document_info"
                        fact_str = f"Document text: {ocr_res['text']}"

            if fact_str:
                if self.memory.is_sensitive_info(fact_str) or self.memory.is_sensitive_info(key):
                    resp = "For security reasons, I cannot store passwords, API keys, or credit card details in personal memory."
                else:
                    success = self.memory.save_memory("personal", key, fact_str)
                    if success:
                        if fact_str.lower().startswith("my ") or fact_str.lower().startswith("that "):
                            resp = f"Got it. I will remember that {fact_str[0].lower() + fact_str[1:]}"
                        else:
                            resp = f"Got it. I will remember that {fact_str}"
                    else:
                        resp = "I couldn't save that."
            else:
                resp = "What information would you like me to save?"

            self.response_manager.add_response(resp, priority=2, force=True)
            print(f"[SAVE] response generated: '{resp}'")
            print(f"[SAVE] completed")
            return resp

        elif intent == "MEMORY_RECALL":
            query = route["params"].get("query", user_transcript)
            recalled = self.memory.recall_memory(query)
            if recalled:
                resp = f"I remember that {recalled[0].lower() + recalled[1:]}" if not recalled.lower().startswith("i ") and not recalled.lower().startswith("my ") else f"{recalled}"
            else:
                resp = "I don't have a specific memory saved for that yet."
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        elif intent == "MEMORY_FORGET":
            key = route["params"].get("key", "")
            success = self.memory.forget_memory(key)
            if success:
                resp = f"Got it. I have deleted that memory about {key}."
            else:
                resp = f"I couldn't find a saved memory matching '{key}'."
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        elif intent == "MEMORY_LIST":
            memories = self.memory.list_all_memories()
            if memories:
                facts = [m["fact_value"] for m in memories[:8]]
                resp = f"I remember {len(memories)} things about you: " + "; ".join(facts) + "."
            else:
                resp = "You haven't asked me to save any personal memories yet."
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        elif intent == "MEMORY_CLEAR":
            count = self.memory.clear_all_memories()
            resp = f"I have cleared all your stored memories ({count} items removed)."
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        # --- FACE MEMORY INTENTS ---
        elif intent == "FACE_REMEMBER":
            raw_cmd = user_transcript
            norm_cmd = user_transcript.strip().lower()
            name = route["params"].get("name")
            key_str = name if name else "active_face"

            print(f"[SAVE] RAW COMMAND: '{raw_cmd}'")
            print(f"[SAVE] NORMALIZED COMMAND: '{norm_cmd}'")
            print(f"[SAVE] INTENT: 'FACE_REMEMBER'")
            print(f"[SAVE] KEY: '{key_str}'")
            print(f"[SAVE] FACT: 'Enroll face as {key_str}'")
            print(f"[SAVE] command received: '{raw_cmd}'")
            print(f"[SAVE] intent detected: 'FACE_REMEMBER'")
            print(f"[SAVE] memory handler started: 'FACE_REMEMBER'")
            if not name or name.lower() in ["this", "face", "this face", "person", "this person", "them", "him", "her", "someone", "friend"]:
                crop = self.face_recognizer.get_primary_face_crop(self.current_frame)
                if crop is None or crop.size == 0:
                    resp = "I couldn't detect a face to save. Please look directly into the camera."
                else:
                    resp = "I can see a face. Whom should I save this face as?"
            else:
                enroll_res = self.face_recognizer.enroll_active_face(self.current_frame, name)
                if enroll_res.get("success"):
                    self.memory.save_memory("relationship", name, f"{name} is saved in face memory.")
                    resp = f"Got it. I have remembered this face as {name}."
                else:
                    resp = enroll_res.get("message", f"I couldn't detect a face to save. Please look directly into the camera so I can remember {name}.")
            self.response_manager.add_response(resp, priority=2, force=True)
            print(f"[SAVE] response generated: '{resp}'")
            print(f"[SAVE] completed")
            return resp

        elif intent == "FACE_FORGET":
            name = route["params"].get("name", "")
            success = self.face_memory.forget_person(name)
            self.memory.forget_memory(name)
            if success:
                resp = f"I have forgotten {name}."
            else:
                resp = f"I don't have any saved face for {name}."
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        elif intent == "FACE_FORGET_ALL":
            count = self.face_memory.forget_all_faces()
            resp = f"I have deleted all saved face profiles ({count} profiles removed)."
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        elif intent == "FACE_LIST":
            people = self.face_memory.list_people()
            if people:
                people_str = ", ".join(people)
                resp = f"I currently remember {len(people)} people: {people_str}."
            else:
                resp = f"I do not have any saved people in face memory."
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        elif intent == "FACE_IDENTIFY":
            if self.last_faces:
                for face in self.last_faces:
                    name = face.get("name")
                    spatial = self.spatial.get_spatial_zone(face["bbox"])
                    if name:
                        # Check long-term memory for relationship fact
                        rel_fact = self.memory.recall_memory(name)
                        if rel_fact:
                            resp = f"There is one person {spatial['full_verbal']}. I recognize him as {name}. ({rel_fact})"
                        else:
                            resp = f"There is one person {spatial['full_verbal']}. I recognize him as {name}."
                        self.response_manager.add_response(resp, priority=2, force=True)
                        return resp
                resp = "There is a person in front of you, but I don't recognize them."
            else:
                resp = "I don't currently see anyone in front of you."
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        # --- OTHER PERCEPTION INTENTS ---
        elif intent == "CURRENCY":
            ocr_res = self.ocr_engine.process_ocr(self.current_frame)
            text_val = ocr_res.get("text", "")
            curr_res = self.currency_detector.analyze_banknote(self.current_frame, detected_text=text_val)
            resp = curr_res["description"]
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        elif intent == "OCR":
            ocr_res = self.ocr_engine.process_ocr(self.current_frame)
            if ocr_res["has_text"]:
                resp = f"The document says: {ocr_res['text']}"
            else:
                resp = "I cannot see clear readable text in the camera view right now."
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        elif intent == "ENVIRONMENT":
            summary = self.scene.generate_scene_summary(self.current_frame, self.last_faces)
            resp = summary["summary"]
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        elif intent == "OBJECT_SEARCH":
            target = route["params"].get("object_name", "object")
            obj_res = self.object_detector.find_target_object(target, self.current_frame)
            resp = obj_res["response_text"]
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        elif intent == "SAFETY":
            if self.last_safety and self.last_safety.get("hazard_detected"):
                resp = self.last_safety["warning_text"]
            else:
                resp = "No immediate physical hazard detected ahead. Please remain cautious."
            self.response_manager.add_response(resp, priority=1, force=True)
            return resp

        elif intent == "SETTINGS":
            setting_key = route["params"]["setting"]
            val = route["params"]["value"]
            self.store.set_setting(setting_key, val)
            if setting_key == "greeting_enabled":
                self.face_recognizer.set_greetings_enabled(val)
                resp = f"Greetings are now {'enabled' if val else 'disabled'}."
            elif setting_key == "environment_monitor_enabled":
                self.monitor.set_mode("continuous" if val else "on_demand")
                resp = f"Continuous environment monitoring is now {'on' if val else 'off'}."
            else:
                resp = "Setting updated."
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        elif intent == "COLOR_IDENTIFY":
            color_res = self.color_detector.detect_dominant_color(self.current_frame)
            resp = color_res["description"]
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        elif intent == "LIGHT_LEVEL_CHECK":
            light_res = self.color_detector.check_ambient_light(self.current_frame)
            resp = light_res["description"]
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        elif intent == "PRODUCT_SCAN":
            ocr_res = self.ocr_engine.process_ocr(self.current_frame)
            ocr_text = ocr_res.get("text", "")
            prod_res = self.product_scanner.scan_product_label(self.current_frame, ocr_text=ocr_text)
            resp = prod_res["description"]
            self.response_manager.add_response(resp, priority=2, force=True)
            return resp

        elif intent == "INTRODUCE":
            resp = OFFICIAL_INTRODUCTION
            self.response_manager.add_response(resp, priority=1, force=True)
            return resp

        elif intent == "SLEEP":
            resp = "Going to sleep mode. Say Hey SG CUBE whenever you need me."
            self.response_manager.add_response(resp, priority=1, force=True)
            return resp

        # General queries fall through to Gemini Live for multimodal reasoning
        return None
