import sqlite3
import tempfile
import unittest
from pathlib import Path

import database


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.original_database = database.DATABASE
        database.DATABASE = Path(self.temporary_directory.name) / "jarvis.db"

    def tearDown(self):
        database.DATABASE = self.original_database
        self.temporary_directory.cleanup()

    def test_fresh_database_has_the_memory_schema(self):
        database.initialize_database()

        connection = sqlite3.connect(database.DATABASE)
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(memories)")
        }
        connection.close()

        self.assertEqual(
            columns,
            {
                "id",
                "content",
                "embedding",
                "memory_type",
                "confidence",
                "created_at",
                "updated_at",
            },
        )

    def test_legacy_database_is_migrated_without_losing_memory(self):
        connection = sqlite3.connect(database.DATABASE)
        connection.execute(
            "CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT NOT NULL)"
        )
        connection.execute("INSERT INTO memories (content) VALUES (?)", ("Learn Python",))
        connection.commit()
        connection.close()

        database.initialize_database()
        memory = database.get_memory(1)

        self.assertEqual(memory[1], "Learn Python")
        self.assertEqual(memory[3], "FACT")
        self.assertEqual(memory[4], 1.0)
        self.assertTrue(memory[5])
        self.assertTrue(memory[6])

    def test_memory_record_lifecycle_preserves_creation_time(self):
        database.initialize_database()
        memory_id = database.add_memory(
            "JARVIS is a local assistant", b"embedding", "PROJECT", 0.9
        )
        original = database.get_memory(memory_id)

        updated = database.update_memory(
            memory_id,
            "JARVIS is a private local assistant",
            b"new-embedding",
            "PROJECT",
            0.95,
        )
        changed = database.get_memory(memory_id)

        self.assertTrue(updated)
        self.assertEqual(changed[1], "JARVIS is a private local assistant")
        self.assertEqual(changed[2], b"new-embedding")
        self.assertEqual(changed[3:5], ("PROJECT", 0.95))
        self.assertEqual(changed[5], original[5])
        self.assertGreaterEqual(changed[6], original[6])


if __name__ == "__main__":
    unittest.main()
