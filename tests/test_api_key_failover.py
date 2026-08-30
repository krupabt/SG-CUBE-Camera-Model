import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assistive.api_key_manager import APIKeyManager

class TestAPIKeyFailover(unittest.TestCase):

    def setUp(self):
        self.pref_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_api_key_prefs"))
        os.makedirs(self.pref_dir, exist_ok=True)
        self.mgr = APIKeyManager(pref_dir=self.pref_dir)

    def tearDown(self):
        import shutil
        if os.path.exists(self.pref_dir):
            shutil.rmtree(self.pref_dir, ignore_errors=True)

    def test_01_single_key_configured(self):
        self.mgr.set_key(1, "AIzaSyTestKey1_ValidFormatForTest123")
        active = self.mgr.get_active_key()
        self.assertIsNotNone(active)
        self.assertEqual(active[0], 1)
        self.assertEqual(active[1], "AIzaSyTestKey1_ValidFormatForTest123")
        self.assertEqual(self.mgr.get_active_key_label(), "Gemini • Key 1")

    def test_02_key1_failover_to_key2(self):
        self.mgr.set_key(1, "AIzaSyTestKey1_ValidFormatForTest123")
        self.mgr.set_key(2, "AIzaSyTestKey2_ValidFormatForTest456")

        # Simulate Key 1 failure
        next_key = self.mgr.get_next_failover_key(1)
        self.assertIsNotNone(next_key)
        self.assertEqual(next_key[0], 2)
        self.assertEqual(next_key[1], "AIzaSyTestKey2_ValidFormatForTest456")
        self.assertEqual(self.mgr.get_active_key_label(), "Gemini • Key 2")

    def test_03_key2_failover_to_key3(self):
        self.mgr.set_key(1, "AIzaSyTestKey1_ValidFormatForTest123")
        self.mgr.set_key(2, "AIzaSyTestKey2_ValidFormatForTest456")
        self.mgr.set_key(3, "AIzaSyTestKey3_ValidFormatForTest789")

        self.mgr.mark_key_failed(1)
        next_key = self.mgr.get_next_failover_key(2)
        self.assertIsNotNone(next_key)
        self.assertEqual(next_key[0], 3)
        self.assertEqual(next_key[1], "AIzaSyTestKey3_ValidFormatForTest789")
        self.assertEqual(self.mgr.get_active_key_label(), "Gemini • Key 3")

    def test_04_all_keys_failed_disconnected_state(self):
        self.mgr.set_key(1, "AIzaSyTestKey1_ValidFormatForTest123")
        self.mgr.set_key(2, "AIzaSyTestKey2_ValidFormatForTest456")
        self.mgr.set_key(3, "AIzaSyTestKey3_ValidFormatForTest789")

        self.mgr.mark_key_failed(1)
        self.mgr.mark_key_failed(2)
        self.mgr.mark_key_failed(3)

        active = self.mgr.get_active_key()
        self.assertIsNone(active)
        self.assertEqual(self.mgr.get_active_key_label(), "Gemini Disconnected")

    def test_05_masked_key_security(self):
        self.mgr.set_key(1, "AIzaSyTestKey1_ValidFormatForTest123")
        masked = self.mgr.get_masked_key(1)
        self.assertTrue(masked.startswith("••••••••••••"))
        self.assertTrue(masked.endswith("t123"))
        self.assertNotIn("AIzaSyTestKey1", masked)

    def test_06_priority_order_persistence(self):
        self.mgr.set_key(1, "AIzaSyTestKey1_ValidFormatForTest123")
        self.mgr.set_key(2, "AIzaSyTestKey2_ValidFormatForTest456")
        self.mgr.set_priority([2, 1, 3])

        # Reload manager from disk
        new_mgr = APIKeyManager(pref_dir=self.pref_dir)
        active = new_mgr.get_active_key()
        self.assertIsNotNone(active)
        self.assertEqual(active[0], 2)
        self.assertEqual(new_mgr.priority, [2, 1, 3])

    def test_07_isolated_connection_test(self):
        ok, msg = self.mgr.test_connection("invalid_short_key")
        self.assertFalse(ok)
        self.assertIn("invalid", msg.lower())

    def test_08_error_classification_invalid_key(self):
        err = Exception("400 INVALID_ARGUMENT: API_KEY_INVALID. Please pass a valid API key.")
        reason, cd = self.mgr.classify_failure(err)
        self.assertEqual(reason, "INVALID_KEY")
        self.assertGreaterEqual(cd, 86400.0)

    def test_09_error_classification_daily_quota(self):
        err = Exception("429 RESOURCE_EXHAUSTED: Daily quota limit reached for GenerateContentRequestsPerDayPerProject")
        reason, cd = self.mgr.classify_failure(err)
        self.assertEqual(reason, "DAILY_QUOTA_EXHAUSTED")
        self.assertGreaterEqual(cd, 86400.0)

    def test_10_error_classification_rate_limit_and_retry_after(self):
        err1 = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded.")
        reason1, cd1 = self.mgr.classify_failure(err1)
        self.assertEqual(reason1, "RATE_LIMIT")
        self.assertEqual(cd1, 60.0)

        err2 = Exception("429 Too Many Requests. Retry-After: 45")
        reason2, cd2 = self.mgr.classify_failure(err2)
        self.assertEqual(reason2, "RATE_LIMIT")
        self.assertEqual(cd2, 45.0)

    def test_11_error_classification_server_error(self):
        err = Exception("503 UNAVAILABLE: The service is temporarily overloaded.")
        reason, cd = self.mgr.classify_failure(err)
        self.assertEqual(reason, "SERVER_ERROR")
        self.assertEqual(cd, 30.0)

    def test_12_error_classification_network_error(self):
        err = Exception("Connection closed prematurely: socket timeout")
        reason, cd = self.mgr.classify_failure(err)
        self.assertEqual(reason, "NETWORK_ERROR")
        self.assertEqual(cd, 15.0)

    def test_13_re_enable_on_key_update(self):
        self.mgr.set_key(1, "AIzaSyTestKey1_ValidFormatForTest123")
        err = Exception("400 API_KEY_INVALID")
        self.mgr.mark_key_failed(1, error=err)
        self.assertIn(1, self.mgr.key_cooldowns)
        self.assertIsNone(self.mgr.get_active_key())

        # Updating key in settings immediately re-enables it
        self.mgr.set_key(1, "AIzaSyTestKey1_CorrectedKeyString999")
        self.assertNotIn(1, self.mgr.key_cooldowns)
        active = self.mgr.get_active_key()
        self.assertIsNotNone(active)
        self.assertEqual(active[0], 1)

if __name__ == "__main__":
    unittest.main()
