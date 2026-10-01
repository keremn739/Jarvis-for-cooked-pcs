import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work.runner import WorkRunner


class FakeProvider:
    instances = []

    def __init__(self, *, cwd=None, on_notification=None):
        self.on_notification = on_notification
        self.thread_id = None
        self.__class__.instances.append(self)

    def start(self):
        pass

    def create_thread(self, *, cwd=None):
        self.thread_id = f"fake-thread-{len(self.instances)}"
        return {"id": self.thread_id}

    def resume_thread(self, thread_id):
        self.thread_id = thread_id
        return {"id": thread_id}

    def send(self, message, *, cwd=None):
        if "Create a concise handover summary" in message:
            yield {
                "method": "item/agentMessage/completed",
                "params": {"item": {"text": "Continue from the existing router implementation."}},
            }
            yield {
                "method": "turn/completed",
                "params": {"turn": {"id": "handover-turn", "status": "completed"}},
            }
            return

        yield {"method": "turn/started", "params": {"turn": {"id": "turn-1"}}}
        yield {"method": "item/agentMessage/delta", "params": {"delta": "hello"}}
        yield {
            "method": "turn/completed",
            "params": {"turn": {"id": "turn-1", "status": "completed"}},
        }

    def stop(self):
        pass


class WorkRunnerTests(unittest.TestCase):
    def setUp(self):
        FakeProvider.instances = []
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

    def _make_run(self, title="Run Codex"):
        project = self.manager.create_project("Runner test")
        task = self.manager.create_task(project["id"], title)
        return self.manager.create_run(task["id"])

    def test_provider_events_become_durable_run_events(self):
        run = self._make_run()
        runner = WorkRunner(self.manager, FakeProvider)

        result = runner.run_codex_turn(run["id"], "Say hello", cwd=self.temp_dir.name)

        self.assertEqual(result["status"], "SUCCEEDED")
        event_types = [event["type"] for event in self.manager.get_events(run["id"])]
        self.assertEqual(
            event_types,
            [
                "RUN_CREATED",
                "RUN_STARTED",
                "TURN_STARTED",
                "AGENT_MESSAGE_DELTA",
                "TURN_COMPLETED",
                "RUN_COMPLETED",
            ],
        )

    def test_rotation_persists_handover_and_replaces_session(self):
        project = self.manager.create_project("Rotation test")
        task = self.manager.create_task(project["id"], "Long task")
        first_run = self.manager.create_run(task["id"])
        runner = WorkRunner(self.manager, FakeProvider, rotation_turns=1)

        runner.run_codex_turn(first_run["id"], "First turn", cwd=self.temp_dir.name)
        first_session = self.manager.get_events(first_run["id"])[-1]["session_id"]
        first_session_row = __import__("work.store", fromlist=["get_agent_session"]).get_agent_session(first_session)

        second_run = self.manager.create_run(task["id"])
        runner.run_codex_turn(second_run["id"], "Continue", cwd=self.temp_dir.name)

        second_session_id = self.manager.get_events(second_run["id"])[-1]["session_id"]
        old_session = __import__("work.store", fromlist=["get_agent_session"]).get_agent_session(first_session)
        new_session = __import__("work.store", fromlist=["get_agent_session"]).get_agent_session(second_session_id)

        self.assertEqual(first_session_row["status"], "COMPLETED")
        self.assertEqual(old_session["status"], "ROTATED")
        self.assertEqual(new_session["status"], "COMPLETED")
        self.assertEqual(new_session["metadata"]["turn_count"], 1)

        events = self.manager.get_events(second_run["id"])
        self.assertIn("HANDOVER_CREATED", [event["type"] for event in events])
        self.assertIn("SESSION_ROTATED", [event["type"] for event in events])

        handovers = [
            event for event in events
            if event["type"] == "HANDOVER_CREATED"
        ]
        self.assertEqual(len(handovers), 1)

        import work.store as store
        artifact = store.get_artifact(handovers[0]["payload"]["artifact_id"])
        self.assertEqual(artifact["metadata"]["kind"], "HANDOVER")
        self.assertIn("Continue from the existing router implementation.", artifact["metadata"]["content"])

    def test_provider_failure_marks_run_failed(self):
        class FailingProvider(FakeProvider):
            def send(self, message, *, cwd=None):
                raise RuntimeError("provider crashed")
                yield

        run = self._make_run("Fail")
        runner = WorkRunner(self.manager, FailingProvider)

        with self.assertRaises(RuntimeError):
            runner.run_codex_turn(run["id"], "Fail")

        failed = self.manager.get_run(run["id"])
        self.assertEqual(failed["status"], "FAILED")
        self.assertEqual(failed["error"], "provider crashed")


if __name__ == "__main__":
    unittest.main()
