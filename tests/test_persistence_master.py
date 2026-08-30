import os
import sys
import time
import unittest
import numpy as np
import cv2
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assistive.memory_store import MemoryStore
from assistive.memory_manager import MemoryManager
from assistive.face_memory import FaceMemory
from assistive.conversation_history import ConversationHistory
from assistive.api_key_manager import APIKeyManager
from visionclaw_gui import SGCubeApp

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "persistence_test_data")

class TestPersistenceMaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        cls.store = MemoryStore(base_dir=TEST_DATA_DIR)
        cls.memory = MemoryManager(db_dir=os.path.join(TEST_DATA_DIR, "memory"))
        cls.face_mem = FaceMemory(storage_dir=os.path.join(TEST_DATA_DIR, "face_memory"))
        cls.history = ConversationHistory(db_dir=os.path.join(TEST_DATA_DIR, "history"))
        cls.key_mgr = APIKeyManager()

    def test_01_user_profile_persistence_and_update(self):
        # Set profile name
        self.store.set_setting("user_name", "Alexth")
        self.store.set_setting("user_display_name", "Alexth")

        # Reload store to simulate restart
        reloaded_store = MemoryStore(base_dir=TEST_DATA_DIR)
        self.assertEqual(reloaded_store.get_setting("user_name"), "Alexth")
        self.assertEqual(reloaded_store.get_setting("user_display_name"), "Alexth")

        # Update profile name to TestUser
        reloaded_store.set_setting("user_display_name", "TestUser")

        # Reload again
        reloaded_store2 = MemoryStore(base_dir=TEST_DATA_DIR)
        self.assertEqual(reloaded_store2.get_setting("user_display_name"), "TestUser")

        # Restore original
        reloaded_store2.set_setting("user_display_name", "Alexth")

    def test_02_face_enrollment_and_recognition_after_restart(self):
        # Create synthetic face crop
        face_crop = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.rectangle(face_crop, (20, 20), (100, 100), (120, 150, 200), -1)

        # Enroll face
        person_id = self.face_mem.save_person("Rahul", face_crop)
        self.assertIsNotNone(person_id)

        # Reload FaceMemory to simulate restart
        reloaded_face_mem = FaceMemory(storage_dir=os.path.join(TEST_DATA_DIR, "face_memory"))
        matched_name, score = reloaded_face_mem.find_match(face_crop)
        self.assertEqual(matched_name, "Rahul")

        # Test unknown face
        unknown_crop = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.circle(unknown_crop, (64, 64), 30, (50, 50, 50), -1)
        unk_name, unk_score = reloaded_face_mem.find_match(unknown_crop, threshold=0.85)
        self.assertNotEqual(unk_name, "Rahul")

    def test_03_face_deletion_persistence(self):
        # Delete enrolled face
        deleted = self.face_mem.forget_person("Rahul")
        self.assertTrue(deleted)

        # Reload to verify deletion persists
        reloaded_face_mem = FaceMemory(storage_dir=os.path.join(TEST_DATA_DIR, "face_memory"))
        self.assertNotIn("Rahul", reloaded_face_mem.list_people())

    def test_04_personal_memory_persistence_and_deletion(self):
        # Save personal memory
        self.memory.save_memory("preference", "favorite_color", "My favorite color is blue.")

        # Query memory
        facts = self.memory.recall_memory("favorite_color")
        self.assertEqual(facts, "My favorite color is blue.")

        # Reload MemoryManager to simulate restart
        reloaded_memory = MemoryManager(db_dir=os.path.join(TEST_DATA_DIR, "memory"))
        reloaded_facts = reloaded_memory.recall_memory("favorite_color")
        self.assertEqual(reloaded_facts, "My favorite color is blue.")

        # Delete memory
        deleted = reloaded_memory.forget_memory("favorite_color")
        self.assertTrue(deleted)

        # Verify deletion persists after restart
        reloaded_memory2 = MemoryManager(db_dir=os.path.join(TEST_DATA_DIR, "memory"))
        self.assertIsNone(reloaded_memory2.recall_memory("favorite_color"))

    def test_05_conversation_history_persistence(self):
        # Create history session
        sid = self.history.create_session("Master Test Session")
        self.history.add_message(sid, "USER", "What is around me?")
        self.history.add_message(sid, "ASSISTANT", "There is a table and a desk in front of you.")

        # Reload ConversationHistory to simulate restart
        reloaded_history = ConversationHistory(db_dir=os.path.join(TEST_DATA_DIR, "history"))
        sessions = reloaded_history.list_all_sessions()
        self.assertTrue(any(s["session_id"] == sid for s in sessions))

        msgs = reloaded_history.get_session_messages(sid)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["sender"].lower(), "user")
        self.assertEqual(msgs[1]["sender"].lower(), "assistant")

    def test_06_history_deletion_isolation(self):
        # Save a memory and profile key first
        self.store.set_setting("user_display_name", "Alexth")
        self.memory.save_memory("preference", "coffee", "I prefer black coffee.")

        # Create history session
        sid = self.history.create_session("Temporary Session")
        self.history.add_message(sid, "USER", "Hello")

        # Delete history session
        deleted = self.history.delete_session(sid)
        self.assertTrue(deleted)

        # Verify history session is gone
        sessions = self.history.list_all_sessions()
        self.assertFalse(any(s["session_id"] == sid for s in sessions))

        # Verify Profile and Personal Memory remain intact!
        self.assertEqual(self.store.get_setting("user_display_name"), "Alexth")
        facts = self.memory.recall_memory("coffee")
        self.assertEqual(facts, "I prefer black coffee.")

if __name__ == "__main__":
    unittest.main()
