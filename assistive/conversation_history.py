import os
import sqlite3
import time
import uuid
import re
import threading
from typing import List, Dict, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_HISTORY_DIR = os.path.join(PROJECT_ROOT, "data", "history")

def is_sensitive_info(text: str) -> bool:
    """ Blocks security credentials, passwords, card numbers, and API keys from being stored """
    if not text:
        return False
    lower = text.lower()
    patterns = [
        r'\b(?:password|passcode|pin|cvv)\b',
        r'\b(?:api[_\s]?key|secret[_\s]?key|access[_\s]?token|auth[_\s]?token)\b',
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        r'\b(?:bank|routing|account)\s*number\b'
    ]
    return any(re.search(p, lower) for p in patterns)

class ConversationHistory:
    """
    High-Performance Persistent SQLite + FTS5 Conversation History Manager.
    Stores user/assistant chat transcripts across application restarts.
    """

    def __init__(self, db_dir: str = DEFAULT_HISTORY_DIR):
        self.db_dir = os.path.abspath(db_dir)
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "conversations.db")

        self.current_turn_chunks: List[str] = []
        self._tls = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._tls, "conn") or self._tls.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA cache_size=-32000;")
            conn.row_factory = sqlite3.Row
            self._tls.conn = conn
        return self._tls.conn

    def _init_db(self):
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        start_timestamp REAL NOT NULL,
                        last_updated REAL NOT NULL,
                        message_count INTEGER DEFAULT 0
                    );
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        sender TEXT NOT NULL,
                        text TEXT NOT NULL,
                        intent TEXT DEFAULT 'GENERAL',
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_sess ON messages (session_id, timestamp);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_upd ON sessions (last_updated DESC);")

                try:
                    cursor.execute("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                            text
                        );
                    """)
                except Exception:
                    pass

                conn.commit()
            print(f"[HISTORY] Database initialized successfully at '{self.db_path}'.")
        except Exception as e:
            print(f"[HISTORY-ERROR] Database init error: {e}")
        finally:
            conn.close()

    def create_session(self, title: Optional[str] = None) -> str:
        now = time.time()
        sid = f"sess_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        if not title:
            title = f"Conversation — {time.strftime('%b %d, %I:%M %p')}"

        try:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    "INSERT INTO sessions (session_id, title, start_timestamp, last_updated, message_count) VALUES (?, ?, ?, ?, ?)",
                    (sid, title, now, now, 0)
                )
                conn.commit()
            print(f"[HISTORY] Created new session '{sid}' ('{title}').")
            return sid
        except Exception as e:
            print(f"[HISTORY-ERROR] Error creating session: {e}")
            return sid

    def log_message(self, session_id: str, sender: str, text: str, intent: str = "GENERAL") -> bool:
        if not text or not text.strip():
            return False

        clean_text = text.strip()
        if is_sensitive_info(clean_text):
            print("[HISTORY] Blocked saving sensitive credentials.")
            return False

        clean_sender = sender.strip().lower()
        now = time.time()
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO messages (session_id, timestamp, sender, text, intent) VALUES (?, ?, ?, ?, ?)",
                    (session_id, now, clean_sender, clean_text, intent)
                )
                cursor.execute(
                    "UPDATE sessions SET last_updated = ?, message_count = message_count + 1 WHERE session_id = ?",
                    (now, session_id)
                )
                try:
                    cursor.execute("INSERT INTO messages_fts (text) VALUES (?)", (clean_text,))
                except Exception:
                    pass
                conn.commit()
            return True
        except Exception as e:
            print(f"[HISTORY-ERROR] Error logging message: {e}")
            return False

    def add_message(self, session_id: str, sender: str, text: str, intent: str = "GENERAL") -> bool:
        """ Alias for log_message for API compatibility """
        return self.log_message(session_id, sender, text, intent)

    def append_turn_chunk(self, chunk_text: str):
        if chunk_text:
            self.current_turn_chunks.append(chunk_text)

    def accumulate_assistant_chunk(self, chunk_text: str):
        """ Alias for append_turn_chunk """
        self.append_turn_chunk(chunk_text)

    def commit_turn(self, session_id: str, sender: str = "assistant", intent: str = "GENERAL") -> Optional[str]:
        if not self.current_turn_chunks:
            return None
        full_text = "".join(self.current_turn_chunks).strip()
        self.current_turn_chunks.clear()
        if full_text:
            self.log_message(session_id, sender, full_text, intent)
            return full_text
        return None

    def finalize_assistant_turn(self, session_id: str, intent: str = "GENERAL") -> Optional[str]:
        """ Alias for commit_turn """
        return self.commit_turn(session_id, sender="assistant", intent=intent)

    def list_all_sessions(self) -> List[Dict]:
        """ Lists all history sessions """
        return self.list_recent_sessions(limit=1000)

    def get_session_messages(self, session_id: str, limit: int = 100) -> List[Dict]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, session_id, timestamp, sender, text, intent FROM messages WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?",
                (session_id, limit)
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[HISTORY-ERROR] Error getting messages: {e}")
            return []

    def get_last_meaningful_message(self, session_id: Optional[str] = None, exclude_roles: Optional[List[str]] = None) -> Optional[str]:
        exclude_roles = exclude_roles or []
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            if session_id:
                cursor.execute(
                    "SELECT text, sender FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT 10",
                    (session_id,)
                )
            else:
                cursor.execute(
                    "SELECT text, sender FROM messages ORDER BY timestamp DESC LIMIT 10"
                )
            rows = cursor.fetchall()
            for r in rows:
                sender = r["sender"]
                text = r["text"].strip()
                if sender not in exclude_roles and len(text) > 2:
                    if not text.lower().startswith("save this") and not text.lower().startswith("remember this"):
                        return text
            return None
        except Exception as e:
            print(f"[HISTORY-ERROR] Error fetching last message: {e}")
            return None

    def list_recent_sessions(self, limit: int = 20) -> List[Dict]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT session_id, title, start_timestamp, last_updated, message_count FROM sessions ORDER BY last_updated DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[HISTORY-ERROR] Error listing sessions: {e}")
            return []

    def search_history(self, keyword: str) -> List[Dict]:
        if not keyword:
            return []
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.id, m.session_id, m.timestamp, m.sender, m.text, m.intent
                FROM messages_fts f
                JOIN messages m ON m.id = f.rowid
                WHERE messages_fts MATCH ?
                LIMIT 50
            """, (f'"{keyword.strip()}"*',))
            rows = cursor.fetchall()
            if rows:
                return [dict(r) for r in rows]
        except Exception:
            pass

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, session_id, timestamp, sender, text, intent FROM messages WHERE text LIKE ? ORDER BY timestamp DESC LIMIT 50",
                (f"%{keyword.strip()}%",)
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[HISTORY-ERROR] Error searching history: {e}")
            return []

    def delete_session(self, session_id: str) -> bool:
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
            return deleted
        except Exception as e:
            print(f"[HISTORY-ERROR] Error deleting session: {e}")
            return False

    def clear_all_history(self) -> int:
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions")
                count = cursor.rowcount
                cursor.execute("DELETE FROM messages")
                try:
                    cursor.execute("DELETE FROM messages_fts")
                except Exception:
                    pass
                conn.commit()
            return count
        except Exception as e:
            print(f"[HISTORY-ERROR] Error clearing history: {e}")
            return 0
