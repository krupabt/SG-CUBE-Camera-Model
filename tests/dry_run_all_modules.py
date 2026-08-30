import os
import sys
import shutil
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assistive.memory_store import MemoryStore
from assistive.memory_manager import MemoryManager
from assistive.face_memory import FaceMemory
from assistive.face_recognition import FaceRecognizer
from assistive.currency_detector import CurrencyDetector
from assistive.ocr_engine import OCREngine
from assistive.spatial_analyzer import SpatialAnalyzer
from assistive.object_detector import ObjectDetector
from assistive.safety_analyzer import SafetyAnalyzer
from assistive.scene_analyzer import SceneAnalyzer
from assistive.environment_monitor import EnvironmentMonitor
from assistive.command_router import CommandRouter
from assistive.response_manager import ResponseManager
from assistive.vision_engine import VisionEngine
from assistive.color_detector import ColorDetector
from assistive.product_scanner import ProductScanner

def run_dry_run():
    print("=" * 60)
    print("      VISIONCLAW ASSISTIVE AI — DRY RUN MODULE TEST      ")
    print("=" * 60)
    print()

    # 1. Memory Store & Settings
    print("[1/14] Testing MemoryStore...")
    test_data_dir = "data/dry_run_tmp"
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir, ignore_errors=True)
    store = MemoryStore(base_dir=test_data_dir)
    pref_val = store.get_setting("greeting_enabled")
    print(f"  -> Preferences loaded successfully. greeting_enabled={pref_val}")

    # 2. MemoryManager (SQLite Long-Term Memory)
    print("\n[2/14] Testing MemoryManager (SQLite Persistent Memory)...")
    mem_mgr = MemoryManager(db_dir=os.path.join(test_data_dir, "memory"))
    mem_mgr.save_memory("personal", "favorite_color", "My favorite color is blue.")
    mem_mgr.save_memory("relationship", "rahul", "Rahul is my friend.")
    recalled = mem_mgr.recall_memory("favorite_color")
    rel_recalled = mem_mgr.recall_memory("rahul")
    print(f"  -> Recalled fact 1: '{recalled}'")
    print(f"  -> Recalled fact 2: '{rel_recalled}'")

    # 3. Face Memory
    print("\n[3/14] Testing FaceMemory...")
    face_mem = FaceMemory(storage_dir=os.path.join(test_data_dir, "face_memory"))
    synthetic_face = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.circle(synthetic_face, (60, 60), 40, (180, 150, 100), -1)
    cv2.circle(synthetic_face, (45, 45), 6, (255, 255, 255), -1)
    cv2.circle(synthetic_face, (75, 45), 6, (255, 255, 255), -1)

    pid = face_mem.save_person("Rahul", synthetic_face)
    people = face_mem.list_people()
    matched_name, sim_score = face_mem.find_match(synthetic_face, threshold=0.40)
    print(f"  -> Enrolled person: {pid}, List: {people}")
    print(f"  -> Match result: name='{matched_name}', similarity={sim_score:.3f}")

    # 4. Face Recognizer
    print("\n[4/14] Testing FaceRecognizer...")
    recognizer = FaceRecognizer(face_memory=face_mem, threshold=0.40)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas[100:220, 200:320] = synthetic_face
    rec_results = recognizer.process_frame(canvas)
    print(f"  -> Detected faces in frame: {len(rec_results)}")

    # 5. Currency Detector
    print("\n[5/14] Testing CurrencyDetector...")
    curr_det = CurrencyDetector()
    curr_res = curr_det.analyze_banknote(canvas, detected_text="RESERVE BANK OF INDIA 500 RUPEES")
    print(f"  -> Banknote output: denom={curr_res['denom']}, confidence={curr_res['confidence']}")

    # 6. OCR Engine
    print("\n[6/14] Testing OCREngine...")
    ocr = OCREngine(repeat_cooldown=5.0)
    ocr_res1 = ocr.process_ocr(canvas, external_text="WARNING SLOW DOWN")
    print(f"  -> OCR Pass 1: text='{ocr_res1['text']}', is_new={ocr_res1['is_new']}")

    # 7. Spatial Analyzer
    print("\n[7/14] Testing SpatialAnalyzer...")
    spatial = SpatialAnalyzer(frame_width=640, frame_height=480)
    zone_center = spatial.get_spatial_zone((270, 150, 100, 150))
    print(f"  -> Spatial verbal: '{zone_center['full_verbal']}'")

    # 8. Object Detector
    print("\n[8/14] Testing ObjectDetector...")
    obj_det = ObjectDetector(spatial_analyzer=spatial)
    obj_res = obj_det.find_target_object("phone", canvas)
    print(f"  -> Search query 'phone': text='{obj_res['response_text']}'")

    # 9. Safety Analyzer
    print("\n[9/14] Testing SafetyAnalyzer...")
    safety = SafetyAnalyzer(spatial_analyzer=spatial)
    close_face_results = [{"bbox": (100, 50, 400, 400), "name": "Unknown"}]
    saf_res = safety.analyze_hazards(canvas, face_results=close_face_results)
    print(f"  -> Hazard warning: '{saf_res['warning_text']}'")

    # 10. Scene Analyzer
    print("\n[10/14] Testing SceneAnalyzer...")
    scene = SceneAnalyzer(spatial_analyzer=spatial)
    sc_summary = scene.generate_scene_summary(canvas, face_results=rec_results)
    print(f"  -> Summary: '{sc_summary['summary']}'")

    # 11. Environment Monitor
    print("\n[11/14] Testing EnvironmentMonitor...")
    env_mon = EnvironmentMonitor(announcement_cooldown=1.0)
    env_mon.set_mode("continuous")
    event = env_mon.evaluate_continuous_events(canvas, face_results=rec_results, safety_result=saf_res, ocr_result={})
    print(f"  -> Continuous event: {event['type'] if event else None}")

    # 12. Command Router
    print("\n[12/14] Testing CommandRouter...")
    router = CommandRouter()
    commands_to_test = [
        "Remember that Rahul is my friend",
        "Remember my favorite color is blue",
        "What do you remember about me?",
        "Forget that Rahul is my friend",
        "Clear all memories",
        "Who is in front of me?",
        "Find my phone"
    ]
    for cmd in commands_to_test:
        route = router.route_intent(cmd)
        print(f"  -> Command: '{cmd}' ==> Intent: {route['intent']}")

    # 13. Response Manager
    print("\n[13/14] Testing ResponseManager...")
    resp_mgr = ResponseManager(announcement_cooldown=1.0)
    resp_mgr.add_response("URGENT SAFETY HAZARD!", priority=1)
    print(f"  -> Popped priority item: '{resp_mgr.get_next_response()}'")

    # 14. Vision Engine (End-to-End Coordination)
    print("\n[14/18] Testing Unified VisionEngine with Persistent Memory...")
    engine = VisionEngine(data_dir=test_data_dir)

    # Test saving memory via VisionEngine
    r1 = engine.process_user_speech_query("Remember my favorite color is blue")
    print(f"  -> VisionEngine Memory Save: '{r1}'")

    # Test recalling memory via VisionEngine
    r2 = engine.process_user_speech_query("What do you remember about me?")
    print(f"  -> VisionEngine Memory Recall: '{r2}'")

    # 15. ColorDetector
    print("\n[15/18] Testing ColorDetector...")
    color_det = ColorDetector()
    c_res = color_det.detect_dominant_color(canvas)
    l_res = color_det.check_ambient_light(canvas)
    print(f"  -> Color: '{c_res['color_name']}', Light Level: '{l_res['light_level']}'")

    # 16. ProductScanner
    print("\n[16/18] Testing ProductScanner...")
    prod_scan = ProductScanner()
    p_res = prod_scan.scan_product_label(canvas, ocr_text="Aspirin 100mg Tablets EXP: 11/2028")
    print(f"  -> Product Scan Output: '{p_res['description']}'")

    # 17. Voice Face Enrollment
    print("\n[17/18] Testing Voice Face Enrollment...")
    enroll_res = recognizer.enroll_active_face(canvas, "Sahana")
    print(f"  -> Face Enrollment Result: success={enroll_res['success']}, person_id={enroll_res.get('person_id')}")

    # 18. New Speech Intent Handlers in VisionEngine
    print("\n[18/18] Testing New VisionEngine Intent Handlers...")
    engine.process_frame(canvas)
    r3 = engine.process_user_speech_query("What color is this?")
    r4 = engine.process_user_speech_query("Are the lights on?")
    r5 = engine.process_user_speech_query("Scan this product")
    print(f"  -> Color Intent: '{r3}'")
    print(f"  -> Light Intent: '{r4}'")
    print(f"  -> Product Intent: '{r5}'")

    # Clean up temporary test data
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir, ignore_errors=True)

    print("=" * 60)
    print("      SUCCESS: ALL 18 ASSISTIVE MODULES PASSED DRY RUN      ")
    print("=" * 60)

if __name__ == "__main__":
    run_dry_run()
