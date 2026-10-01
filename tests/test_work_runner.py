import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work.runner import WorkRunner


class FakeProvider:
    def __init__(self, *, cwd=None, on_notification=None):
        self.on_notification = on_notification

    def start(self):
        pass

    def create_thread(self, *, cwd=None):
        return {"id": "fake-thread"}

    def send(self, message, *, cwd=None):
        for event in (
            {"method": "turn/started", "params": {"turn": {"id": "turn-1"}}},
            {"method": "item/agentMessage/delta", "params": {"delta": "hello"}},
            {"method": "turn/completed", "params": {"turn": {"id": "turn-1", "status": "completed"}}},
        ):
            yield event

    def stop(self):
        pass


class WorkRunnerTests(unittest.TestCase):
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

    def test_provider_events_become_durable_run_events(self):
        project = self.manager.create_project("Runner test")
        task = self.manager.create_task(project["id"], "Run Codex")
        run = self.manager.create_run(task["id"])
        runner = WorkRunner(self.manager, FakeProvider)

        result = runner.run_codex_turn(run["id"], "Say hello", cwd=self.temp_dir.name)

        self.assertEqual(result["status"], "SUCCEEDED")
        event_types = [event["type"] for event in self.manager.get_events(run["id"])]
        self.assertEqual(event_types, ["RUN_CREATED", "RUN_STARTED", "TURN_STARTED", "AGENT_MESSAGE_DELTA", "TURN_COMPLETED", "RUN_COMPLETED"])

    def test_provider_failure_marks_run_failed(self):
        class FailingProvider(FakeProvider):
            def send(self, message, *, cwd=None):
                raise RuntimeError("provider crashed")
                yield

        project = self.manager.create_project("Failure test")
        task = self.manager.create_task(project["id"], "Fail")
        run = self.manager.create_run(task["id"])
        runner = WorkRunner(self.manager, FailingProvider)

        with self.assertRaises(RuntimeError):
            runner.run_codex_turn(run["id"], "Fail")

        failed = self.manager.get_run(run["id"])
        self.assertEqual(failed["status"], "FAILED")
        self.assertEqual(failed["error"], "provider crashed")


if __name__ == "__main__":
    unittest.main()
