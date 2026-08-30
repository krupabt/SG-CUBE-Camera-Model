import os
import sqlite3
import time
import re
import threading
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MEMORY_DIR = os.path.join(PROJECT_ROOT, "data", "memory")

SENSITIVE_KEYWORDS = [
    "password", "passcode", "api key", "apikey", "secret key",
    "credit card", "debit card", "cvv", "social security", "ssn",
    "auth token", "access token", "pin number", "banking credential"
]

class MemoryManager:
    """
    High-Performance SQLite + FTS5 + RAM-Cached Persistent Long-Term Memory System.
    Stores personal facts, preferences, user relationships across sessions.
    Data is stored locally in data/memory/memories.db with instant in-memory RAM caching
    and SQLite FTS5 full-text indexing for sub-millisecond retrieval.
    """

    def __init__(self, db_dir: str = None):
        if db_dir is None or db_dir in ["data/memory", "memory"]:
            self.db_dir = DEFAULT_MEMORY_DIR
        else:
            self.db_dir = os.path.abspath(db_dir)

        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "memories.db")

        # Thread-safe in-memory RAM Cache
        self._ram_cache: Dict[str, str] = {}
        self._all_memories_cache: Optional[List[Dict]] = None
        self._cache_lock = threading.RLock()
        self._tls = threading.local()

        self._init_database()
        self._warm_cache()

    def is_sensitive_info(self, text: str) -> bool:
        """ Returns True if the text contains security credentials or passwords """
        if not text:
            return False
        low = text.lower()
        return any(kw in low for kw in SENSITIVE_KEYWORDS)

    def _get_connection(self) -> sqlite3.Connection:
        """ Returns a reusable thread-local SQLite connection with WAL & high-performance PRAGMAs """
        if not hasattr(self._tls, "conn") or self._tls.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.execute("PRAGMA cache_size=-64000;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            self._tls.conn = conn
        return self._tls.conn

    def _init_database(self):
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT NOT NULL,
                        key_phrase TEXT UNIQUE NOT NULL,
                        fact_value TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_key ON memories (key_phrase);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_cat ON memories (category);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_upd ON memories (updated_at DESC);")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_date TEXT NOT NULL,
                        summary_text TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );
                """)

                try:
                    cursor.execute("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                            key_phrase,
                            fact_value
                        );
                    """)
                except Exception as fe:
                    print(f"[MEMORY] [WARN] FTS5 initialization fallback: {fe}")

                conn.commit()
        except Exception as e:
            print(f"[MEMORY] [ERROR] DB Init failed: {e}")
        finally:
            conn.close()

    def _warm_cache(self):
        """ Pre-loads stored keys into the in-memory RAM cache on startup """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT key_phrase, fact_value FROM memories")
            rows = cursor.fetchall()
            with self._cache_lock:
                for k, v in rows:
                    self._ram_cache[k.strip().lower()] = v
        except Exception as e:
            print(f"[MEMORY] [WARN] Cache warming failed: {e}")

    def save_memory(self, category: str, key_phrase: str, fact_value: str) -> bool:
        """
        Saves or updates a persistent memory entry in SQLite database with immediate RAM cache update.
        """
        if not key_phrase or not fact_value:
            print("[SAVE] ERROR: Empty key or fact value.")
            return False

        if self.is_sensitive_info(key_phrase) or self.is_sensitive_info(fact_value):
            print("[SAVE] ERROR: Blocked saving sensitive security credential.")
            return False

        clean_key = key_phrase.strip().lower()
        clean_val = fact_value.strip()
        now = time.time()

        print(f"[SAVE] database path: {self.db_path}")
        print(f"[SAVE] database opened")
        print(f"[SAVE] INSERT started")
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO memories (category, key_phrase, fact_value, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(key_phrase) DO UPDATE SET
                        fact_value=excluded.fact_value,
                        updated_at=excluded.updated_at
                """, (category, clean_key, clean_val, now, now))

                # Update FTS5 Table
                try:
                    cursor.execute("DELETE FROM memories_fts WHERE key_phrase = ?", (clean_key,))
                    cursor.execute("INSERT INTO memories_fts (key_phrase, fact_value) VALUES (?, ?)", (clean_key, clean_val))
                except Exception:
                    pass

                conn.commit()

            print("[SAVE] INSERT completed")
            print("[SAVE] COMMIT completed")

            # Verification in same connection
            print("[SAVE] SELECT verification")
            cursor.execute("SELECT key_phrase, fact_value FROM memories WHERE key_phrase = ?", (clean_key,))
            row = cursor.fetchone()
            if row:
                print(f"[SAVE] memory verified: Key='{row[0]}', Value='{row[1]}'")
            else:
                print(f"[SAVE] WARNING: Memory row not found immediately after commit.")

            # Update RAM Cache immediately
            with self._cache_lock:
                self._ram_cache[clean_key] = clean_val
                norm_alias = clean_key.replace("_", " ").replace("colour", "color")
                self._ram_cache[norm_alias] = clean_val
                self._all_memories_cache = None

            print("[SAVE] completed")
            return True
        except Exception as e:
            print(f"[SAVE] ERROR: {e}")
            return False

    def recall_memory(self, query: str) -> Optional[str]:
        """
        High-Performance Hierarchical Memory Retrieval:
        1. Thread-safe RAM Cache Lookup (< 0.001 ms)
        2. Exact Indexed SQL Key Lookup (< 0.1 ms)
        3. Normalized Key & Alias Resolution (< 0.2 ms)
        4. SQLite FTS5 Full-Text Search Match (< 0.3 ms)
        5. Fast word-intersection fallback
        """
        if not query:
            return None

        raw_query = query.strip().lower().replace("?", "").replace("'", "").replace("’", "")
        clean_search = re.sub(r'^(?:what is|whats|do you know|do you remember|tell me|who is|can you tell me)\s+(?:my|the|a|an)?\s*', '', raw_query).strip()
        print(f"[MEMORY] [RECALL] Querying persistent memories for: '{query}' (cleaned: '{clean_search}')")

        # 1. RAM Cache Lookup (Sub-microsecond)
        with self._cache_lock:
            if clean_search in self._ram_cache:
                val = self._ram_cache[clean_search]
                print(f"[MEMORY] [RECALL] RAM Cache match! Key='{clean_search}', Value='{val}'")
                return val
            if raw_query in self._ram_cache:
                val = self._ram_cache[raw_query]
                print(f"[MEMORY] [RECALL] RAM Cache match! Key='{raw_query}', Value='{val}'")
                return val

        conn = self._get_connection()
        cursor = conn.cursor()

        # 2. Exact Indexed SQL Key Lookup
        try:
            cursor.execute("SELECT fact_value FROM memories WHERE key_phrase = ? LIMIT 1", (clean_search,))
            row = cursor.fetchone()
            if row:
                val = row[0]
                with self._cache_lock:
                    self._ram_cache[clean_search] = val
                print(f"[MEMORY] [RECALL] Exact Indexed SQL match! Key='{clean_search}', Value='{val}'")
                return val
        except Exception:
            pass

        # 3. Normalized Key & Alias Resolution
        norm_key_search = clean_search.replace("colour", "color").replace("pet", "dog").replace("profession", "job")
        if norm_key_search != clean_search:
            with self._cache_lock:
                if norm_key_search in self._ram_cache:
                    val = self._ram_cache[norm_key_search]
                    print(f"[MEMORY] [RECALL] Normalized Alias match! Key='{norm_key_search}', Value='{val}'")
                    return val
            try:
                cursor.execute("SELECT fact_value FROM memories WHERE key_phrase = ? LIMIT 1", (norm_key_search,))
                row = cursor.fetchone()
                if row:
                    val = row[0]
                    with self._cache_lock:
                        self._ram_cache[clean_search] = val
                    return val
            except Exception:
                pass

        # 4. SQLite FTS5 Full-Text Search
        try:
            fts_tokens = [w for w in clean_search.split() if len(w) >= 3 and w not in ["what", "whats", "where", "who", "when", "which", "how", "your", "know", "remember", "about", "tell"]]
            if fts_tokens:
                fts_query = " OR ".join(f'"{t}"*' for t in fts_tokens)
                cursor.execute("SELECT fact_value FROM memories_fts WHERE memories_fts MATCH ? LIMIT 1", (fts_query,))
                row = cursor.fetchone()
                if row:
                    val = row[0]
                    with self._cache_lock:
                        self._ram_cache[clean_search] = val
                    print(f"[MEMORY] [RECALL] FTS5 match found! Query='{fts_query}', Value='{val}'")
                    return val
        except Exception:
            pass

        # 5. In-Memory Substring / Word Intersection Fallback
        try:
            cursor.execute("SELECT key_phrase, fact_value FROM memories ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            if not rows:
                return None

            for key_p, val_p in rows:
                norm_key = key_p.strip().lower().replace("_", " ").replace("'", "").replace("’", "")
                if len(norm_key) >= 2 and (norm_key in raw_query or norm_key in clean_search or clean_search in norm_key):
                    with self._cache_lock:
                        self._ram_cache[clean_search] = val_p
                    print(f"[MEMORY] [RECALL] Key match found! Key='{key_p}', Value='{val_p}'")
                    return val_p

            query_words = [w for w in clean_search.split() if len(w) >= 3 and w not in ["what", "whats", "where", "who", "when", "which", "how", "your", "know", "remember", "about"]]
            for key_p, val_p in rows:
                norm_key = key_p.strip().lower()
                for qw in query_words:
                    if qw == norm_key or qw in norm_key or norm_key in qw:
                        with self._cache_lock:
                            self._ram_cache[clean_search] = val_p
                        print(f"[MEMORY] [RECALL] Word match found! Word='{qw}', Key='{key_p}', Value='{val_p}'")
                        return val_p
        except Exception as e:
            print(f"[MEMORY] [ERROR] Recall error: {e}")

        print(f"[MEMORY] [RECALL] No matching memory found for query: '{query}'")
        return None

    def search_memories(self, keyword: str) -> List[Dict]:
        """ Searches all memories containing the given keyword via FTS5 / SQL """
        results = []
        if not keyword:
            return results

        clean_kw = keyword.strip().lower()
        conn = self._get_connection()
        cursor = conn.cursor()

        # Try FTS5 Search first
        try:
            cursor.execute("SELECT key_phrase, fact_value FROM memories_fts WHERE memories_fts MATCH ? LIMIT 50", (f'"{clean_kw}"*',))
            rows = cursor.fetchall()
            if rows:
                for r in rows:
                    results.append({
                        "id": 1,
                        "category": "personal",
                        "key_phrase": r[0],
                        "fact_value": r[1],
                        "created_at": time.time()
                    })
                return results
        except Exception:
            pass

        # SQL LIKE Fallback
        try:
            like_kw = f"%{clean_kw}%"
            cursor.execute("""
                SELECT id, category, key_phrase, fact_value, created_at
                FROM memories
                WHERE key_phrase LIKE ? OR fact_value LIKE ?
                ORDER BY updated_at DESC
            """, (like_kw, like_kw))
            rows = cursor.fetchall()
            for r in rows:
                results.append({
                    "id": r[0],
                    "category": r[1],
                    "key_phrase": r[2],
                    "fact_value": r[3],
                    "created_at": r[4]
                })
        except Exception as e:
            print(f"[MEMORY] [ERROR] Search error: {e}")

        return results

    def list_all_memories(self) -> List[Dict]:
        """ Lists all stored persistent memories with RAM caching """
        with self._cache_lock:
            if self._all_memories_cache is not None:
                return list(self._all_memories_cache)

        results = []
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, category, key_phrase, fact_value, created_at
                FROM memories
                ORDER BY updated_at DESC
            """)
            rows = cursor.fetchall()
            for r in rows:
                results.append({
                    "id": r[0],
                    "category": r[1],
                    "key_phrase": r[2],
                    "fact_value": r[3],
                    "created_at": r[4]
                })
            with self._cache_lock:
                self._all_memories_cache = results
        except Exception as e:
            print(f"[MEMORY] [ERROR] List error: {e}")

        return results

    def forget_memory(self, key_phrase: str) -> bool:
        """ Deletes a memory by key phrase or keyword match with immediate cache invalidation """
        if not key_phrase:
            return False

        clean_key = key_phrase.strip().lower()
        print(f"[MEMORY] [DELETE] Deleting memory key: '{clean_key}'")
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memories WHERE key_phrase = ?", (clean_key,))
                deleted = cursor.rowcount > 0

                if not deleted:
                    cursor.execute("DELETE FROM memories WHERE key_phrase LIKE ? OR fact_value LIKE ?", (f"%{clean_key}%", f"%{clean_key}%"))
                    deleted = cursor.rowcount > 0

                try:
                    cursor.execute("DELETE FROM memories_fts WHERE key_phrase = ?", (clean_key,))
                except Exception:
                    pass

                conn.commit()

            if deleted:
                with self._cache_lock:
                    self._ram_cache.pop(clean_key, None)
                    self._ram_cache.pop(clean_key.replace("_", " "), None)
                    self._all_memories_cache = None
                print(f"[MEMORY] [DELETE] Delete successful for key: '{clean_key}'")
            return deleted
        except Exception as e:
            print(f"[MEMORY] [ERROR] Delete error: {e}")
            return False

    def clear_all_memories(self) -> int:
        """ Deletes all stored persistent memories with complete cache clear """
        print("[MEMORY] [DELETE] Clearing all stored memories...")
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memories")
                deleted_count = cursor.rowcount
                try:
                    cursor.execute("DELETE FROM memories_fts")
                except Exception:
                    pass
                conn.commit()

            with self._cache_lock:
                self._ram_cache.clear()
                self._all_memories_cache = None

            print(f"[MEMORY] [DELETE] Cleared {deleted_count} items from memories database.")
            return deleted_count
        except Exception as e:
            print(f"[MEMORY] [ERROR] Clear all error: {e}")
            return 0

    def get_relevant_user_context(self) -> str:
        """ Returns a clean summary string of stored facts/relationships for Gemini context """
        memories = self.list_all_memories()
        if not memories:
            return ""

        facts = [m["fact_value"] for m in memories[:15]]
        context_str = "User Stored Facts & Preferences: " + "; ".join(facts) + "."
        print(f"[MEMORY] [LOAD] Loaded {len(memories)} memories into session context.")
        return context_str
