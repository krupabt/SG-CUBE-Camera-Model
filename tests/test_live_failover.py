import os
import time
import shutil
import tempfile
import unittest
from assistive.api_key_manager import APIKeyManager

class TestLiveFailover(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='test_failover_')
        self.mgr = APIKeyManager(pref_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_key1_invalid_fails_over_to_key2(self):
        # Key 1 invalid, Key 2 valid, Key 3 valid
        self.mgr.set_key(1, 'AIzaSyInvalidKeyOne1234567890')
        self.mgr.set_key(2, 'AIzaSyValidKeyTwo123456789012')
        self.mgr.set_key(3, 'AIzaSyValidKeyThree12345678901')

        active = self.mgr.get_active_key()
        self.assertEqual(active[0], 1)

        # Simulate Gemini Live connection error on Key 1
        fake_err = Exception('API key not valid. Please pass a valid API key.')
        next_key = self.mgr.get_next_failover_key(1, error=fake_err)
        
        self.assertIsNotNone(next_key)
        self.assertEqual(next_key[0], 2)
        self.assertEqual(next_key[1], 'AIzaSyValidKeyTwo123456789012')

    def test_key1_and_key2_invalid_fails_over_to_key3(self):
        self.mgr.set_key(1, 'AIzaSyInvalidKeyOne1234567890')
        self.mgr.set_key(2, 'AIzaSyInvalidKeyTwo1234567890')
        self.mgr.set_key(3, 'AIzaSyValidKeyThree12345678901')

        err1 = Exception('403 Forbidden: API_KEY_INVALID')
        self.mgr.get_next_failover_key(1, error=err1)

        err2 = Exception('401 Unauthorized: Invalid API key')
        next_key = self.mgr.get_next_failover_key(2, error=err2)

        self.assertIsNotNone(next_key)
        self.assertEqual(next_key[0], 3)
        self.assertEqual(next_key[1], 'AIzaSyValidKeyThree12345678901')

    def test_all_keys_invalid_terminates_cleanly_without_infinite_loop(self):
        self.mgr.set_key(1, 'AIzaSyInvalidKeyOne1234567890')
        self.mgr.set_key(2, 'AIzaSyInvalidKeyTwo1234567890')
        self.mgr.set_key(3, 'AIzaSyInvalidKeyThree12345678')

        err = Exception('API_KEY_INVALID')
        self.mgr.get_next_failover_key(1, error=err)
        self.mgr.get_next_failover_key(2, error=err)
        final_key = self.mgr.get_next_failover_key(3, error=err)

        self.assertIsNone(final_key)
        self.assertIsNone(self.mgr.get_active_key())

    def test_transient_network_error_classification(self):
        net_err = Exception('ConnectError: Failed to connect to host: timeout')
        reason, cd = self.mgr.classify_failure(net_err)
        self.assertEqual(reason, 'NETWORK_ERROR')
        self.assertEqual(cd, 15.0)

    def test_quota_exhaustion_classification(self):
        quota_err = Exception('429 ResourceExhausted: Daily quota exceeded')
        reason, cd = self.mgr.classify_failure(quota_err)
        self.assertEqual(reason, 'DAILY_QUOTA_EXHAUSTED')
        self.assertEqual(cd, 86400.0)

if __name__ == '__main__':
    unittest.main()
