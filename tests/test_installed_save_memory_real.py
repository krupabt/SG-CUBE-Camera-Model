import os
import sys
import sqlite3
import time
import unittest
import numpy as np

# Ensure installed application modules are imported
INSTALLED_APP_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Programs", "SG-CUBE")
sys.path.insert(0, INSTALLED_APP_DIR)

from assistive.vision_engine import VisionEngine
from assistive.memory_manager import MemoryManager
from assistive.command_router import CommandRouter

class TestInstalledSaveMemoryReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Point to installed application's real database directory
        cls.engine = VisionEngine(data_dir=os.path.join(INSTALLED_APP_DIR, "data"))
        cls.db_path = cls.engine.memory.db_path
        print(f"\n[TEST] Active Installed DB Path: {cls.db_path}")
        self_check_conn = sqlite3.connect(cls.db_path)
        cur = self_check_conn.cursor()
        cur.execute("PRAGMA journal_mode;")
        cls.journal_mode = cur.fetchone()[0]
        cur.execute("PRAGMA busy_timeout;")
        cls.busy_timeout = cur.fetchone()[0]
        self_check_conn.close()

    def _query_db_fact(self, key_phrase):
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT key_phrase, fact_value, updated_at FROM memories WHERE key_phrase = ?", (key_phrase.lower(),))
            return cur.fetchone()
        finally:
            conn.close()

    def test_01_sqlite_pragmas(self):
        """ Verify SQLite WAL mode and busy timeout in installed application """
        print(f"[TEST 01] SQLite journal_mode={self.journal_mode}, busy_timeout={self.busy_timeout}")
        self.assertEqual(self.journal_mode.lower(), "wal")
        self.assertGreaterEqual(self.busy_timeout, 5000)

    def test_02_remember_favorite_color_blue(self):
        """ Test: 'Remember my favorite color is blue.' -> Intent, Handler, DB Insert, Commit, Record Verification """
        cmd = "Remember my favorite color is blue."
        resp = self.engine.process_user_speech_query(cmd)
        self.assertIsNotNone(resp)
        self.assertIn("blue", resp.lower())

        # Inspect SQLite DB directly
        row = self._query_db_fact("favorite color")
        self.assertIsNotNone(row, "Record for 'favorite color' must exist in SQLite database!")
        self.assertEqual(row[0], "favorite color")
        self.assertEqual(row[1], "My favorite color is blue.")
        print(f"[TEST 02] Verified in DB: key='{row[0]}', fact='{row[1]}'")

    def test_03_remember_favorite_fruit_apple(self):
        """ Test: 'Remember my favorite fruit is apple.' -> DB Insert & Recall Verification """
        cmd = "Remember my favorite fruit is apple."
        resp = self.engine.process_user_speech_query(cmd)
        self.assertIsNotNone(resp)
        self.assertIn("apple", resp.lower())

        row = self._query_db_fact("favorite fruit")
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "favorite fruit")
        self.assertEqual(row[1], "My favorite fruit is apple.")

        # Test Recall
        recall_resp = self.engine.process_user_speech_query("What is my favorite fruit?")
        self.assertIsNotNone(recall_resp)
        self.assertIn("apple", recall_resp.lower())
        print(f"[TEST 03] Recall Verified: '{recall_resp}'")

    def test_04_contextual_save_this_information_with_history(self):
        """ Test: Prior turn + 'Save this information.' -> Resolves and persists to DB """
        sess_id = self.engine.history.create_session()
        self.engine.history.add_message(sess_id, "user", "My dentist appointment is at 4pm on Thursday")
        self.engine.history.add_message(sess_id, "assistant", "I noted that your dentist appointment is at 4pm on Thursday.")

        cmd = "Save this information."
        resp = self.engine.process_user_speech_query(cmd, session_id=sess_id)
        self.assertIsNotNone(resp)
        self.assertIn("thursday", resp.lower())

        # Direct SQLite verify
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT fact_value FROM memories WHERE fact_value LIKE '%Thursday%'")
        row = cur.fetchone()
        conn.close()
        self.assertIsNotNone(row)
        print(f"[TEST 04] Contextual Save Verified in DB: '{row[0]}'")

    def test_05_bare_save_this_without_context(self):
        """ Test: Bare 'Save this.' without prior context in fresh session -> Prompts clearly without freezing """
        sess_empty = self.engine.history.create_session()
        resp = self.engine.process_user_speech_query("Save this.", session_id=sess_empty)
        self.assertIsNotNone(resp)
        self.assertIn("what information would you like me to save", resp.lower())
        print(f"[TEST 05] Prompt response: '{resp}'")

    def test_06_save_this_face_without_face(self):
        """ Test: 'Save this face.' when no face is visible -> Reports clear message without freezing """
        self.engine.current_frame = np.zeros((480, 640, 3), dtype=np.uint8) # blank frame
        resp = self.engine.process_user_speech_query("Save this face.")
        self.assertIsNotNone(resp)
        self.assertIn("couldn't detect a face", resp.lower())
        print(f"[TEST 06] Face save response: '{resp}'")

    def test_07_save_this_face_with_name_and_face(self):
        """ Test: 'Save this face as Rahul.' with simulated face crop """
        face_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        self.engine.current_frame = face_img
        self.engine.face_recognizer.enroll_active_face = lambda frame, name: {"success": True, "message": f"Enrolled {name}"}

        resp = self.engine.process_user_speech_query("Save this face as Rahul.")
        self.assertIsNotNone(resp)
        self.assertIn("rahul", resp.lower())

        row = self._query_db_fact("rahul")
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "rahul")
        print(f"[TEST 07] Face enrolled into relationship memory: '{row[1]}'")

    def test_08_delete_and_forget_memory(self):
        """ Test: 'Forget my favorite fruit' -> Deletes from DB and confirms """
        resp = self.engine.process_user_speech_query("Forget my favorite fruit")
        self.assertIsNotNone(resp)
        self.assertIn("deleted", resp.lower())

        # Direct DB verify
        row = self._query_db_fact("favorite fruit")
        self.assertIsNone(row, "Memory for 'favorite fruit' should be completely deleted!")
        print(f"[TEST 08] Memory successfully deleted from SQLite DB.")

    def test_09_persistence_after_restart(self):
        """ Test: Facts saved persist across re-instantiation of VisionEngine """
        self.engine.process_user_speech_query("Remember that my dog name is Bruno.")
        
        # Simulate restart
        new_engine = VisionEngine(data_dir=os.path.join(INSTALLED_APP_DIR, "data"))
        recalled = new_engine.memory.recall_memory("dog name")
        self.assertIsNotNone(recalled)
        self.assertIn("bruno", recalled.lower())
        print(f"[TEST 09] Persisted across restart: '{recalled}'")


if __name__ == "__main__":
    unittest.main()
