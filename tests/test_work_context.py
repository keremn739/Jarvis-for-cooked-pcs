import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work.context import build_run_context, render_provider_context
from work.specs import approve_spec, create_spec


class WorkContextTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "jarvis.db"
        self.patch = patch("database.DATABASE", self.database_path)
        self.patch.start()
        import work.store as store
        store.DATABASE = self.database_path
        store.initialize_work_schema()
        from work.manager import WorkManager
        self.manager = WorkManager()

    def tearDown(self):
        self.patch.stop()
        self.temp_dir.cleanup()

    def test_context_contains_project_task_run_and_current_approval(self):
        project = self.manager.create_project("Jarvis", "Personal assistant", "/tmp/jarvis")
        task = self.manager.create_task(project["id"], "Build agent layer", "Implement persistent work")
        run = self.manager.create_run(task["id"])
        artifact = create_spec(project["id"], "plan.md", "Implement provider abstraction", run_id=run["id"], artifact_type="PLAN")
        approve_spec(run["id"], artifact["id"])

        context = build_run_context(run["id"])

        self.assertEqual(context["project"]["name"], "Jarvis")
        self.assertEqual(context["task"]["title"], "Build agent layer")
        self.assertEqual(context["run"]["id"], run["id"])
        self.assertEqual(len(context["approved_artifacts"]), 1)
        self.assertIn("Implement provider abstraction", render_provider_context(context))

    def test_stale_approval_is_not_passed_to_provider(self):
        project = self.manager.create_project("Jarvis")
        task = self.manager.create_task(project["id"], "Build")
        run = self.manager.create_run(task["id"])
        artifact = create_spec(project["id"], "plan.md", "Version A", run_id=run["id"], artifact_type="PLAN")
        approval = approve_spec(run["id"], artifact["id"])

        import work.store as store
        connection = store._connect()
        connection.execute("UPDATE artifacts SET content_hash = ? WHERE id = ?", ("stale", artifact["id"]))
        connection.commit()
        connection.close()

        context = build_run_context(run["id"])
        self.assertFalse(context["approved_artifacts"])
        self.assertFalse(approval["status"] == "PENDING")


if __name__ == "__main__":
    unittest.main()
