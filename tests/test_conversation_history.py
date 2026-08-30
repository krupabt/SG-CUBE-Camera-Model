import os
import shutil
import unittest
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assistive.conversation_history import ConversationHistory, is_sensitive_info

class TestConversationHistory(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.abspath(f"data/test_history_db_tmp_{uuid.uuid4().hex[:6]}")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
        self.history = ConversationHistory(db_dir=self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_create_session_and_add_messages(self):
        sid = self.history.create_session("Test Session 1")
        self.assertTrue(sid.startswith("sess_"))

        # Add user & assistant messages
        saved_user = self.history.add_message(sid, "user", "What is around me?")
        self.assertTrue(saved_user)

        self.history.accumulate_assistant_chunk("There are two ")
        self.history.accumulate_assistant_chunk("people in front of you.")
        final_resp = self.history.finalize_assistant_turn(sid)

        self.assertEqual(final_resp, "There are two people in front of you.")

        # Verify messages retrieved
        msgs = self.history.get_session_messages(sid)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["sender"], "user")
        self.assertEqual(msgs[1]["sender"], "assistant")

    def test_persistence_across_instances(self):
        sid = self.history.create_session("Persistence Test")
        self.history.add_message(sid, "user", "Find my keys.")
        self.history.add_message(sid, "assistant", "Your keys are on the desk.")

        # Instantiate second ConversationHistory pointing to same test_dir (simulating app restart)
        history2 = ConversationHistory(db_dir=self.test_dir)
        sessions = history2.list_all_sessions()
        self.assertEqual(len(sessions), 1)

        msgs = history2.get_session_messages(sid)
        self.assertEqual(len(msgs), 2)
        self.assertIn("keys", msgs[0]["text"])

    def test_search_history(self):
        sid1 = self.history.create_session("Session Alpha")
        self.history.add_message(sid1, "user", "Who is in front of me?")
        self.history.add_message(sid1, "assistant", "I recognize Rahul.")

        sid2 = self.history.create_session("Session Beta")
        self.history.add_message(sid2, "user", "How much money is this?")
        self.history.add_message(sid2, "assistant", "This is a 500 rupee banknote.")

        results = self.history.search_history("Rahul")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_id"], sid1)

        results_currency = self.history.search_history("500")
        self.assertEqual(len(results_currency), 1)
        self.assertEqual(results_currency[0]["session_id"], sid2)

    def test_delete_single_and_clear_all(self):
        sid1 = self.history.create_session("Session One")
        sid2 = self.history.create_session("Session Two")

        self.assertEqual(len(self.history.list_all_sessions()), 2)

        # Delete single session
        deleted = self.history.delete_session(sid1)
        self.assertTrue(deleted)
        self.assertEqual(len(self.history.list_all_sessions()), 1)

        # Clear all history
        cleared_count = self.history.clear_all_history()
        self.assertEqual(cleared_count, 1)
        self.assertEqual(len(self.history.list_all_sessions()), 0)

    def test_security_credential_filtering(self):
        self.assertTrue(is_sensitive_info("My password is Secret123!"))
        self.assertTrue(is_sensitive_info("API_KEY=AIzaSyA12345678"))

        sid = self.history.create_session("Security Test")
        saved = self.history.add_message(sid, "user", "Save my credit card 4532-1234-5678-9012")
        self.assertFalse(saved)

if __name__ == "__main__":
    unittest.main()
