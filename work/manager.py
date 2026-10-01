"""Application-level orchestration for durable JARVIS work state."""

from . import store


class WorkManager:
    """Own projects, tasks, runs, sessions, and their event history.

    This layer is intentionally provider-agnostic. Codex, a future local agent,
    or another provider can attach to an AgentSession without changing the
    durable work model.
    """

    def create_project(self, name, description="", root_path=None):
        project = store.create_project(name, description, root_path)
        return project

    def create_task(self, project_id, title, description="", priority=0):
        task = store.create_task(project_id, title, description, priority)
        return task

    def create_run(self, task_id, metadata=None):
        run = store.create_run(task_id, metadata)
        store.append_event(run["id"], "RUN_CREATED", {"task_id": task_id})
        return store.get_run(run["id"])

    def start_run(self, run_id):
        run = store.update_run(run_id, status="RUNNING")
        store.append_event(run_id, "RUN_STARTED")
        return run

    def create_session(self, run_id, provider):
        session = store.create_agent_session(run_id, provider)
        return session

    def attach_provider_session(self, session_id, provider_session_id):
        return store.update_agent_session(
            session_id,
            status="ACTIVE",
            provider_session_id=provider_session_id,
        )

    def record_event(self, run_id, event_type, payload=None, session_id=None):
        return store.append_event(run_id, event_type, payload, session_id)

    def complete_run(self, run_id):
        run = store.update_run(run_id, status="SUCCEEDED", finished_at=store._utc_now())
        store.append_event(run_id, "RUN_COMPLETED")
        return run

    def fail_run(self, run_id, error):
        run = store.update_run(
            run_id,
            status="FAILED",
            error=str(error),
            finished_at=store._utc_now(),
        )
        store.append_event(run_id, "RUN_FAILED", {"error": str(error)})
        return run

    def cancel_run(self, run_id):
        run = store.update_run(
            run_id,
            status="CANCELLED",
            finished_at=store._utc_now(),
        )
        store.append_event(run_id, "RUN_CANCELLED")
        return run

    def get_run(self, run_id):
        return store.get_run(run_id)

    def get_events(self, run_id):
        return store.list_events(run_id)
