import unittest
import os
import shutil
import tempfile
import time
import threading
import numpy as np
import cv2

from assistive.vision_engine import VisionEngine
from assistive.command_router import CommandRouter
from assistive.memory_manager import MemoryManager
from assistive.face_memory import FaceMemory
from assistive.conversation_history import ConversationHistory


class TestSaveCommandMaster(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sgcube_save_test_")
        self.engine = VisionEngine(data_dir=self.test_dir)
        self.router = CommandRouter()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_remember_simple_fact(self):
        """ Test: Remember my favorite color is blue """
        cmd = "Remember my favorite color is blue"
        resp = self.engine.process_user_speech_query(cmd)
        self.assertIsNotNone(resp)
        self.assertIn("favorite color is blue", resp.lower())

        # Verify in database
        recalled = self.engine.memory.recall_memory("What is my favorite color?")
        self.assertIsNotNone(recalled)
        self.assertIn("blue", recalled.lower())

    def test_02_remember_multi_word_fact(self):
        """ Test: Remember that Rahul is my best friend from college """
        cmd = "Remember that Rahul is my best friend from college"
        resp = self.engine.process_user_speech_query(cmd)
        self.assertIsNotNone(resp)
        self.assertIn("rahul", resp.lower())

        # Recall
        recalled = self.engine.memory.recall_memory("Who is Rahul?")
        self.assertIsNotNone(recalled)
        self.assertIn("best friend", recalled.lower())

    def test_03_save_information_explicit(self):
        """ Test: Save this: doctor appointment on Friday at 10am """
        cmd = "Save this: doctor appointment on Friday at 10am"
        resp = self.engine.process_user_speech_query(cmd)
        self.assertIsNotNone(resp)

        recalled = self.engine.memory.recall_memory("doctor appointment")
        self.assertIsNotNone(recalled)
        self.assertIn("friday", recalled.lower())

    def test_04_save_this_contextual_from_history(self):
        """ Test: User discusses fact, then says 'Save this' """
        sid = self.engine.history.create_session("Test Session")
        self.engine.history.add_message(sid, "user", "My sister birthday is on October 5th")
        self.engine.history.add_message(sid, "assistant", "I noted that your sister birthday is October 5th.")

        cmd = "Save this"
        resp = self.engine.process_user_speech_query(cmd)
        self.assertIsNotNone(resp)
        self.assertIn("october 5th", resp.lower())

        # Verify saved in long-term memory
        recalled = self.engine.memory.recall_memory("sister birthday")
        self.assertIsNotNone(recalled)
        self.assertIn("october 5th", recalled.lower())

    def test_05_save_this_information_bare_prompt(self):
        """ Test: 'Save this information' when no prior history exists """
        cmd = "Save this information"
        resp = self.engine.process_user_speech_query(cmd)
        self.assertIsNotNone(resp)
        self.assertIn("what information", resp.lower())

    def test_06_save_this_face_with_name(self):
        """ Test: Save this face as Rahul with a synthetic face frame """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.ellipse(frame, (320, 240), (80, 110), 0, 0, 360, (180, 200, 240), -1)
        self.engine.current_frame = frame

        cmd = "Save this face as Rahul"
        resp = self.engine.process_user_speech_query(cmd)
        self.assertIsNotNone(resp)
        self.assertIn("rahul", resp.lower())

        people = self.engine.face_memory.list_people()
        self.assertIn("Rahul", people)

    def test_07_save_this_face_no_face_visible(self):
        """ Test: 'Save this face' with blank/black frame """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.engine.current_frame = frame

        cmd = "Save this face as Alex"
        resp = self.engine.process_user_speech_query(cmd)
        self.assertIsNotNone(resp)
        self.assertIn("couldn't detect a face", resp.lower())

    def test_08_save_this_face_bare_with_face_visible(self):
        """ Test: Bare 'Save this face' with face visible """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.ellipse(frame, (320, 240), (80, 110), 0, 0, 360, (180, 200, 240), -1)
        self.engine.current_frame = frame

        cmd = "Save this face"
        resp = self.engine.process_user_speech_query(cmd)
        self.assertIsNotNone(resp)
        self.assertTrue("whom should i save" in resp.lower() or "see a face" in resp.lower())

    def test_09_duplicate_save_no_corruption(self):
        """ Test: Repeating save multiple times does not corrupt or duplicate """
        cmd = "Remember my favorite food is pizza"
        for _ in range(5):
            resp = self.engine.process_user_speech_query(cmd)
            self.assertIsNotNone(resp)

        all_mems = self.engine.memory.list_all_memories()
        matching = [m for m in all_mems if "pizza" in m["fact_value"].lower()]
        self.assertEqual(len(matching), 1, "Duplicate saves should maintain exactly one memory entry.")

    def test_10_concurrent_database_access(self):
        """ Test: 50 concurrent threads writing and reading memories without database lock errors """
        errors = []

        def worker(idx):
            try:
                self.engine.memory.save_memory("test", f"thread_key_{idx}", f"Fact value from thread {idx}")
                recalled = self.engine.memory.recall_memory(f"thread_key_{idx}")
                if not recalled:
                    errors.append(f"Thread {idx} recall failed")
            except Exception as e:
                errors.append(f"Thread {idx} exception: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent database errors: {errors}")

    def test_11_persistence_after_restart(self):
        """ Test: Save memory, close engine, instantiate new engine from same dir, recall memory """
        cmd = "Remember my favorite movie is Interstellar"
        self.engine.process_user_speech_query(cmd)

        new_engine = VisionEngine(data_dir=self.test_dir)
        recalled = new_engine.memory.recall_memory("What is my favorite movie?")
        self.assertIsNotNone(recalled)
        self.assertIn("interstellar", recalled.lower())

    def test_12_security_sensitive_data_blocked(self):
        """ Test: Sensitive credentials like password or api key are blocked """
        cmd = "Remember my password is SecretPassword123"
        resp = self.engine.process_user_speech_query(cmd)
        self.assertIsNotNone(resp)
        self.assertIn("security reasons", resp.lower())

        all_mems = self.engine.memory.list_all_memories()
        self.assertEqual(len(all_mems), 0)


if __name__ == "__main__":
    unittest.main()
