import os
import json
import time
import logging
from typing import Dict, Any

DEFAULT_PREFERENCES: Dict[str, Any] = {
    "voice_enabled": True,
    "greeting_enabled": True,
    "greeting_cooldown_seconds": 30.0,
    "recognition_threshold": 0.55,
    "environment_monitor_enabled": False,
    "safety_alerts_enabled": True,
    "announcement_cooldown_seconds": 10.0,
    "ocr_language": "en",
    "currency_mode": "INR",
    "response_verbosity": "concise",
    "assistant_voice": "Puck",
    "last_greeting_date": "",
    "last_greeting_timestamp": 0.0,
    "first_run_completed": False,
    "user_name": "",
    "user_display_name": "",
    "user_profile_photo_path": "",
    "user_profile_notes": ""
}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

class MemoryStore:
    def __init__(self, base_dir: str = None):
        if base_dir is None or base_dir == "data":
            self.base_dir = DEFAULT_DATA_DIR
        else:
            self.base_dir = os.path.abspath(base_dir)

        self.face_dir = os.path.join(self.base_dir, "face_memory")
        self.pref_dir = os.path.join(self.base_dir, "user_preferences")
        self.logs_dir = os.path.join(self.base_dir, "logs")

        self._ensure_dirs()
        self.pref_file = os.path.join(self.pref_dir, "preferences.json")
        self.preferences = self.load_preferences()
        self._setup_logging()

    def _ensure_dirs(self):
        for path in [self.base_dir, self.face_dir, self.pref_dir, self.logs_dir]:
            os.makedirs(path, exist_ok=True)

    def _setup_logging(self):
        log_path = os.path.join(self.logs_dir, "visionclaw.log")
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        self.logger = logging.getLogger("VisionClaw")

    def load_preferences(self) -> Dict[str, Any]:
        if os.path.exists(self.pref_file):
            try:
                with open(self.pref_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    prefs = DEFAULT_PREFERENCES.copy()
                    prefs.update(saved)
                    return prefs
            except Exception as e:
                print(f"Error loading preferences: {e}")
        return DEFAULT_PREFERENCES.copy()

    def save_preferences(self, prefs: Dict[str, Any] = None):
        if prefs is not None:
            self.preferences.update(prefs)
        try:
            with open(self.pref_file, "w", encoding="utf-8") as f:
                json.dump(self.preferences, f, indent=2)
        except Exception as e:
            print(f"Error saving preferences: {e}")

    def get_setting(self, key: str, default: Any = None) -> Any:
        return self.preferences.get(key, default if default is not None else DEFAULT_PREFERENCES.get(key))

    def set_setting(self, key: str, value: Any):
        self.preferences[key] = value
        self.save_preferences()

    def should_trigger_startup_greeting(self) -> bool:
        """
        Determines whether a time-aware startup greeting should be spoken.
        Prevents repetitive greetings on background reconnects.
        """
        today_str = time.strftime("%Y-%m-%d")
        last_date = self.get_setting("last_greeting_date", "")
        last_ts = self.get_setting("last_greeting_timestamp", 0.0)
        now = time.time()

        # Fire if a new day or if more than 3 hours have passed since last startup greeting
        if last_date != today_str or (now - last_ts) > 10800.0:
            self.set_setting("last_greeting_date", today_str)
            self.set_setting("last_greeting_timestamp", now)
            return True
        return False
