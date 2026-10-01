import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class WorkManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "jarvis.db"
        self.patch = patch("database.DATABASE", self.database_path)
        self.patch.start()

        # Import after patching so the work store uses the isolated database.
        import work.store as store
        store.DATABASE = self.database_path
        store.initialize_work_schema()
        self.store = store

        from work.manager import WorkManager
        self.manager = WorkManager()

    def tearDown(self):
        self.patch.stop()
        self.temp_dir.cleanup()

    def test_project_task_run_and_session_lifecycle(self):
        project = self.manager.create_project(
            "Test Project",
            "A durable work-manager test",
            "/tmp/test-project",
        )
        task = self.manager.create_task(project["id"], "Implement feature")
        run = self.manager.create_run(task["id"], {"source": "test"})
        session = self.manager.create_session(run["id"], "codex")
        attached = self.manager.attach_provider_session(session["id"], "thread-123")

        self.assertEqual(project["status"], "ACTIVE")
        self.assertEqual(task["status"], "PLANNED")
        self.assertEqual(run["status"], "QUEUED")
        self.assertEqual(attached["provider_session_id"], "thread-123")
        self.assertEqual(attached["status"], "ACTIVE")

        self.manager.start_run(run["id"])
        self.manager.record_event(
            run["id"],
            "TOOL_COMPLETED",
            {"tool": "pytest", "success": True},
            session["id"],
        )
        completed = self.manager.complete_run(run["id"])

        self.assertEqual(completed["status"], "SUCCEEDED")
        events = self.manager.get_events(run["id"])
        self.assertEqual(
            [event["type"] for event in events],
            ["RUN_CREATED", "RUN_STARTED", "TOOL_COMPLETED", "RUN_COMPLETED"],
        )
        self.assertEqual(events[2]["payload"]["tool"], "pytest")

    def test_failed_run_preserves_error(self):
        project = self.manager.create_project("Test Project")
        task = self.manager.create_task(project["id"], "Failing task")
        run = self.manager.create_run(task["id"])

        failed = self.manager.fail_run(run["id"], "tests failed")

        self.assertEqual(failed["status"], "FAILED")
        self.assertEqual(failed["error"], "tests failed")
        self.assertEqual(self.manager.get_events(run["id"])[-1]["type"], "RUN_FAILED")

    def test_provider_session_id_is_optional_until_provider_starts(self):
        project = self.manager.create_project("Test Project")
        task = self.manager.create_task(project["id"], "Provider task")
        run = self.manager.create_run(task["id"])
        session = self.manager.create_session(run["id"], "codex")

        self.assertIsNone(session["provider_session_id"])
        self.assertEqual(session["status"], "STARTING")


if __name__ == "__main__":
    unittest.main()
