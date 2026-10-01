"""State-machine helpers for runs and agent sessions."""

from __future__ import annotations

from . import store


_RUN_TRANSITIONS = {
    "QUEUED": {"RUNNING", "CANCELLED"},
    "RUNNING": {"WAITING_FOR_USER", "SUCCEEDED", "FAILED", "TIMED_OUT", "CANCELLED"},
    "WAITING_FOR_USER": {"RUNNING", "CANCELLED", "FAILED"},
    "SUCCEEDED": set(),
    "FAILED": {"RUNNING"},
    "TIMED_OUT": {"RUNNING"},
    "CANCELLED": set(),
}

_SESSION_TRANSITIONS = {
    "STARTING": {"ACTIVE", "FAILED", "CANCELLED"},
    "ACTIVE": {"WAITING", "COMPLETED", "FAILED", "CANCELLED"},
    "WAITING": {"ACTIVE", "COMPLETED", "FAILED", "CANCELLED"},
    "COMPLETED": set(),
    "FAILED": {"STARTING"},
    "CANCELLED": set(),
}


def transition_run(run_id, target_status, *, error=None):
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"Run does not exist: {run_id}")
    target = target_status.upper()
    if target not in _RUN_TRANSITIONS[run["status"]]:
        raise ValueError(f"Invalid run transition: {run['status']} -> {target}")

    started_at = None
    finished_at = None
    if target == "RUNNING" and run["started_at"] is None:
        started_at = store._utc_now()
    if target in {"SUCCEEDED", "FAILED", "TIMED_OUT", "CANCELLED"}:
        finished_at = store._utc_now()

    updated = store.update_run(
        run_id,
        status=target,
        error=error,
        started_at=started_at,
        finished_at=finished_at,
    )
    event_map = {
        "RUNNING": "RUN_STARTED",
        "WAITING_FOR_USER": "APPROVAL_REQUIRED",
        "SUCCEEDED": "RUN_COMPLETED",
        "FAILED": "RUN_FAILED",
        "TIMED_OUT": "RUN_FAILED",
        "CANCELLED": "RUN_CANCELLED",
    }
    event_type = event_map.get(target)
    if event_type:
        payload = {"status": target}
        if error:
            payload["error"] = str(error)
        store.append_event(run_id, event_type, payload)
    return updated


def transition_session(session_id, target_status):
    session = store.get_agent_session(session_id)
    if session is None:
        raise ValueError(f"Agent session does not exist: {session_id}")
    target = target_status.upper()
    if target not in _SESSION_TRANSITIONS[session["status"]]:
        raise ValueError(f"Invalid session transition: {session['status']} -> {target}")
    return store.update_agent_session(session_id, status=target)
