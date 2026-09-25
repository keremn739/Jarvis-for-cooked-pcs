import sqlite3
import tempfile
import threading
import time
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import database
import interaction_history


class InteractionHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.original_database = database.DATABASE
        database.DATABASE = Path(self.temporary_directory.name) / "jarvis.db"
        database.initialize_database()

    def tearDown(self):
        database.DATABASE = self.original_database
        self.temporary_directory.cleanup()

    def test_persists_only_message_and_timestamp_in_separate_table(self):
        database.add_interaction("Find the weather")
        rows = database.get_interactions()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "Find the weather")
        self.assertTrue(rows[0][2])
        connection = sqlite3.connect(database.DATABASE)
        columns = [row[1] for row in connection.execute(
            "PRAGMA table_info(interaction_history)"
        )]
        connection.close()
        self.assertEqual(columns, ["id", "message", "created_at"])
        self.assertEqual(database.get_memories(), [])

    def test_redacts_sensitive_matches_and_attached_credential_values(self):
        safe = interaction_history.redact_sensitive_content(
            "Email me at user@example.com; password is hunter2"
        )

        self.assertNotIn("user@example.com", safe)
        self.assertNotIn("hunter2", safe)
        self.assertIn("[REDACTED]", safe)

    def test_automatic_enqueue_persists_message(self):
        writer = interaction_history.InteractionHistoryWriter()
        writer.enqueue("Summarize this document")
        writer._queue.join()

        self.assertEqual(database.get_interactions()[0][1], "Summarize this document")

    def test_enqueue_does_not_wait_for_slow_writer(self):
        started = threading.Event()
        release = threading.Event()

        def slow_write(_message):
            started.set()
            release.wait(2)

        writer = interaction_history.InteractionHistoryWriter(write=slow_write)
        start = time.monotonic()
        writer.enqueue("A completed request")
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 0.1)
        self.assertTrue(started.wait(1))
        release.set()
        writer._queue.join()

    def test_close_drains_queued_writes_before_stopping_worker(self):
        written = []
        writer = interaction_history.InteractionHistoryWriter(
            write=written.append
        )
        writer.enqueue("First completed request")
        writer.enqueue("Second completed request")

        writer.close()

        self.assertEqual(
            written,
            ["First completed request", "Second completed request"],
        )
        self.assertFalse(writer._thread.is_alive())
        writer.close()  # Closing is safe to call more than once.

    def test_main_enqueues_only_after_interaction_finishes(self):
        events = []
        source = Path(__file__).resolve().parents[1] / "main.py"
        with patch("builtins.input", side_effect=["hello", "exit"]), \
             patch("builtins.print", side_effect=lambda *args, **kwargs: events.append("print")), \
             patch("router.route_message", side_effect=lambda _message: events.append("route") or {"steps": []}), \
             patch("interaction_history.enqueue_interaction", side_effect=lambda _message: events.append("log")), \
             patch("interaction_history.close_interaction_history", side_effect=lambda: events.append("close")):
            runpy.run_path(str(source), run_name="__main__")

        self.assertLess(events.index("route"), events.index("log"))
        self.assertEqual(events.count("log"), 1)
        self.assertLess(events.index("log"), events.index("close"))


if __name__ == "__main__":
    unittest.main()
