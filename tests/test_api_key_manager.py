import os
import sys
import json
import base64
import shutil
import unittest
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assistive.api_key_manager import APIKeyManager

class TestAPIKeyManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_key_mgr_")
        self.mgr = APIKeyManager(pref_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_empty_and_masked_representation(self):
        # Unset env and keys for test isolation
        old_env = os.environ.pop("GEMINI_API_KEY", None)
        self.mgr.keys = {1: "", 2: "", 3: ""}
        self.mgr.active_key_num = None
        try:
            self.assertIsNone(self.mgr.load_api_key())
            self.assertEqual(self.mgr.get_masked_key(1), "Not Configured")
        finally:
            if old_env:
                os.environ["GEMINI_API_KEY"] = old_env

    def test_save_and_load_api_key(self):
        test_key = "AIzaSyDummyTestKeyForUnitTesting12345"
        self.assertTrue(self.mgr.save_api_key(test_key))

        loaded = self.mgr.load_api_key()
        self.assertEqual(loaded, test_key)
        self.assertEqual(self.mgr.get_masked_key(), "••••••••••••2345")

    def test_remove_api_key(self):
        test_key = "AIzaSyDummyTestKeyForUnitTesting67890"
        self.mgr.save_api_key(test_key)
        self.assertEqual(self.mgr.load_api_key(), test_key)

        self.assertTrue(self.mgr.remove_api_key())
        self.assertEqual(self.mgr.get_masked_key(), "Not Configured")

    def test_stored_file_never_contains_the_plaintext_key(self):
        test_key = "AIzaSyDummyTestKeyForUnitTesting12345"
        self.mgr.save_api_key(test_key)
        with open(self.mgr.cred_file, "r", encoding="utf-8") as f:
            raw = f.read()
        self.assertNotIn(test_key, raw)

    @unittest.skipUnless(os.name == "nt", "DPAPI is Windows-only")
    def test_keys_are_dpapi_encrypted_not_merely_encoded(self):
        try:
            import win32crypt  # noqa: F401
        except Exception:
            self.skipTest("pywin32 not installed")

        test_key = "AIzaSyDummyTestKeyForUnitTesting12345"
        self.mgr.save_api_key(test_key)
        with open(self.mgr.cred_file, "r", encoding="utf-8") as f:
            stored = json.load(f)["keys"]["1"]

        self.assertTrue(stored.startswith("dpapi:"))
        # The old scheme was reversible by anyone with the file. This asserts the
        # regression cannot come back: base64-decoding alone must not reveal the key.
        self.assertNotIn(test_key.encode("utf-8"),
                         base64.b64decode(stored[len("dpapi:"):].encode("utf-8")))
        # ...and it still round-trips through a freshly constructed manager.
        self.assertEqual(APIKeyManager(pref_dir=self.test_dir).load_api_key(), test_key)

    def test_legacy_base64_file_still_loads(self):
        test_key = "AIzaSyLegacyBase64KeyForUnitTest98765"
        legacy = {
            "keys": {"1": base64.b64encode(test_key.encode("utf-8")).decode("utf-8"),
                     "2": "", "3": ""},
            "priority": [1, 2, 3],
        }
        with open(self.mgr.cred_file, "w", encoding="utf-8") as f:
            json.dump(legacy, f)

        mgr = APIKeyManager(pref_dir=self.test_dir)
        self.assertEqual(mgr.keys[1], test_key)

        if os.name == "nt":
            try:
                import win32crypt  # noqa: F401
            except Exception:
                return
            # Loading an old file upgrades it in place, so the plaintext-recoverable
            # form does not survive the next launch.
            with open(mgr.cred_file, "r", encoding="utf-8") as f:
                self.assertTrue(json.load(f)["keys"]["1"].startswith("dpapi:"))

    def test_validation_invalid_format(self):
        ok, msg = self.mgr.test_connection("short")
        self.assertFalse(ok)
        self.assertIn("invalid", msg.lower())

if __name__ == "__main__":
    unittest.main()
