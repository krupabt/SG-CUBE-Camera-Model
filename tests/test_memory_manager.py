import os
import shutil
import unittest
from assistive.memory_manager import MemoryManager

class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = "data/test_memory_db_tmp"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
        self.memory = MemoryManager(db_dir=self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_recall_memory(self):
        # Save personal fact
        saved = self.memory.save_memory("personal", "favorite_color", "My favorite color is blue.")
        self.assertTrue(saved)

        # Recall fact
        recalled = self.memory.recall_memory("favorite color")
        self.assertIsNotNone(recalled)
        self.assertIn("blue", recalled)

    def test_relationship_memory(self):
        self.memory.save_memory("relationship", "rahul", "Rahul is my friend.")
        rel = self.memory.recall_memory("who is rahul")
        self.assertIsNotNone(rel)
        self.assertIn("Rahul is my friend", rel)

    def test_forget_memory(self):
        self.memory.save_memory("personal", "coffee", "I prefer black coffee.")
        self.assertIsNotNone(self.memory.recall_memory("coffee"))

        forgot = self.memory.forget_memory("coffee")
        self.assertTrue(forgot)
        self.assertIsNone(self.memory.recall_memory("coffee"))

    def test_clear_all_memories(self):
        self.memory.save_memory("fact", "fact1", "Fact 1")
        self.memory.save_memory("fact", "fact2", "Fact 2")
        self.assertEqual(len(self.memory.list_all_memories()), 2)

        cleared = self.memory.clear_all_memories()
        self.assertEqual(cleared, 2)
        self.assertEqual(len(self.memory.list_all_memories()), 0)

    def test_security_secret_blocking(self):
        # Should refuse to store passwords or API keys
        saved = self.memory.save_memory("security", "my password", "Password123!")
        self.assertFalse(saved)

    def test_persistence_across_reconnects(self):
        # Save memory in instance 1
        self.memory.save_memory("personal", "name", "My name is Alexth.")

        # Instantiate instance 2 (simulating app restart)
        memory_instance2 = MemoryManager(db_dir=self.test_dir)
        recalled = memory_instance2.recall_memory("what's my name?")

        self.assertIsNotNone(recalled)
        self.assertIn("Alexth", recalled)

if __name__ == "__main__":
    unittest.main()
