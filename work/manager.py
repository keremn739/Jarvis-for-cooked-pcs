"""Application-level orchestration for durable JARVIS work state."""

from . import lifecycle, store


class WorkManager:
    """Own projects, tasks, runs, sessions, and their event history.

    Provider adapters never own durable application state. They attach to an
    AgentSession and report events through this manager.
    """

    def create_project(self, name, description="", root_path=None):
        return store.create_project(name, description, root_path)

    def create_task(self, project_id, title, description="", priority=0):
        return store.create_task(project_id, title, description, priority)

    def create_run(self, task_id, metadata=None):
        run = store.create_run(task_id, metadata)
        store.append_event(run["id"], "RUN_CREATED", {"task_id": task_id})
        return store.get_run(run["id"])

    def start_run(self, run_id):
        return lifecycle.transition_run(run_id, "RUNNING")

    def create_session(self, run_id, provider):
        return store.create_agent_session(run_id, provider)

    def attach_provider_session(self, session_id, provider_session_id):
        return store.update_agent_session(
            session_id,
            status="ACTIVE",
            provider_session_id=provider_session_id,
        )

    def transition_session(self, session_id, status):
        return lifecycle.transition_session(session_id, status)

    def record_event(self, run_id, event_type, payload=None, session_id=None):
        return store.append_event(run_id, event_type, payload, session_id)

    def complete_run(self, run_id):
        return lifecycle.transition_run(run_id, "SUCCEEDED")

    def fail_run(self, run_id, error):
        return lifecycle.transition_run(run_id, "FAILED", error=str(error))

    def cancel_run(self, run_id):
        return lifecycle.transition_run(run_id, "CANCELLED")

    def get_run(self, run_id):
        return store.get_run(run_id)

    def get_events(self, run_id):
        return store.list_events(run_id)
