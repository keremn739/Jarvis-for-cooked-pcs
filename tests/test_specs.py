import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work.specs import approval_is_current, approve_spec, content_hash, create_spec


class SpecApprovalTests(unittest.TestCase):
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

    def test_approval_is_bound_to_exact_content_hash(self):
        project = self.manager.create_project("Spec test")
        task = self.manager.create_task(project["id"], "Build feature")
        run = self.manager.create_run(task["id"])
        artifact = create_spec(project["id"], "plan.md", "Implement feature A", run_id=run["id"], artifact_type="PLAN")
        approval = approve_spec(run["id"], artifact["id"])

        self.assertEqual(artifact["content_hash"], content_hash("Implement feature A"))
        self.assertTrue(approval_is_current(approval["id"]))

        # Simulate the artifact being edited after approval.
        import work.store as store
        connection = store._connect()
        connection.execute(
            "UPDATE artifacts SET content_hash = ? WHERE id = ?",
            (content_hash("Implement feature B"), artifact["id"]),
        )
        connection.commit()
        connection.close()

        self.assertFalse(approval_is_current(approval["id"]))

    def test_only_specs_and_plans_can_be_approved(self):
        project = self.manager.create_project("Approval test")
        task = self.manager.create_task(project["id"], "Build")
        run = self.manager.create_run(task["id"])
        artifact = create_spec(project["id"], "output.log", "output", run_id=run["id"], artifact_type="OUTPUT")

        with self.assertRaises(ValueError):
            approve_spec(run["id"], artifact["id"])


if __name__ == "__main__":
    unittest.main()
