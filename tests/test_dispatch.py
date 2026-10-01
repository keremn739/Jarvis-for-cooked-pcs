import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work.dispatch import dispatch_work


class FakeRunner:
    def __init__(self, manager):
        self.manager = manager
        self.calls = []

    def run_codex_turn(self, run_id, message, *, cwd=None):
        self.calls.append((run_id, message, cwd))
        return self.manager.get_run(run_id)


class DispatchTests(unittest.TestCase):
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

    def test_dispatch_creates_project_task_and_run(self):
        runner = FakeRunner(self.manager)
        result = dispatch_work("Implement feature", project_root=self.temp_dir.name, manager=self.manager, runner=runner)
        self.assertEqual(result["status"], "QUEUED")
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(runner.calls[0][1], "Implement feature")
        self.assertEqual(runner.calls[0][2], str(Path(self.temp_dir.name).resolve()))


if __name__ == "__main__":
    unittest.main()
