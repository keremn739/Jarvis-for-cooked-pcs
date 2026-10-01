"""SQLite persistence for the JARVIS work-management domain."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from database import DATABASE
from .models import APPROVAL_STATUSES, ARTIFACT_TYPES, EVENT_TYPES, PROJECT_STATUSES, RUN_STATUSES, SESSION_STATUSES, TASK_STATUSES


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _id():
    return str(uuid.uuid4())


def _connect():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_work_schema():
    connection = _connect()
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', root_path TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE, title TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', status TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE, status TEXT NOT NULL, started_at TEXT, finished_at TEXT, error TEXT, metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS agent_sessions (id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, provider TEXT NOT NULL, provider_session_id TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL, last_activity TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, session_id TEXT REFERENCES agent_sessions(id) ON DELETE SET NULL, type TEXT NOT NULL, sequence INTEGER NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, UNIQUE(run_id, sequence));
        CREATE TABLE IF NOT EXISTS artifacts (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE, run_id TEXT REFERENCES runs(id) ON DELETE SET NULL, type TEXT NOT NULL, name TEXT NOT NULL, path TEXT, content_hash TEXT, metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS approvals (id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, artifact_id TEXT REFERENCES artifacts(id) ON DELETE SET NULL, status TEXT NOT NULL, content_hash TEXT, approved_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
        CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_run ON agent_sessions(run_id);
        CREATE INDEX IF NOT EXISTS idx_events_run_sequence ON events(run_id, sequence);
        CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project_id);
        CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals(run_id);
    """)
    connection.commit()
    connection.close()


def _validate(value, allowed, field):
    value = str(value).upper()
    if value not in allowed:
        raise ValueError(f"Unsupported {field}: {value}")
    return value


def create_project(name, description="", root_path=None, status="ACTIVE"):
    status = _validate(status, PROJECT_STATUSES, "project status")
    now, project_id = _utc_now(), _id()
    connection = _connect()
    connection.execute("INSERT INTO projects (id, name, description, root_path, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (project_id, name, description, root_path, status, now, now))
    connection.commit(); connection.close()
    return get_project(project_id)


def get_project(project_id):
    connection = _connect(); row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone(); connection.close()
    return dict(row) if row else None


def get_project_by_root(root_path):
    connection = _connect(); row = connection.execute("SELECT * FROM projects WHERE root_path = ? ORDER BY updated_at DESC LIMIT 1", (str(root_path),)).fetchone(); connection.close()
    return dict(row) if row else None


def list_projects(status=None):
    connection = _connect()
    rows = connection.execute("SELECT * FROM projects ORDER BY created_at").fetchall() if status is None else connection.execute("SELECT * FROM projects WHERE status = ? ORDER BY created_at", (_validate(status, PROJECT_STATUSES, "project status"),)).fetchall()
    connection.close(); return [dict(row) for row in rows]


def create_task(project_id, title, description="", priority=0, status="PLANNED"):
    status = _validate(status, TASK_STATUSES, "task status")
    now, task_id = _utc_now(), _id(); connection = _connect()
    try:
        connection.execute("INSERT INTO tasks (id, project_id, title, description, status, priority, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (task_id, project_id, title, description, status, int(priority), now, now)); connection.commit()
    except sqlite3.IntegrityError as error:
        connection.rollback(); raise ValueError(f"Project does not exist: {project_id}") from error
    finally: connection.close()
    return get_task(task_id)


def get_task(task_id):
    connection = _connect(); row = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone(); connection.close(); return dict(row) if row else None


def create_run(task_id, metadata=None, status="QUEUED"):
    status = _validate(status, RUN_STATUSES, "run status")
    now, run_id = _utc_now(), _id(); connection = _connect()
    try:
        connection.execute("INSERT INTO runs (id, task_id, status, metadata_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)", (run_id, task_id, status, json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True), now, now)); connection.commit()
    except sqlite3.IntegrityError as error:
        connection.rollback(); raise ValueError(f"Task does not exist: {task_id}") from error
    finally: connection.close()
    return get_run(run_id)


def get_run(run_id):
    connection = _connect(); row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone(); connection.close()
    if not row: return None
    result = dict(row); result["metadata"] = json.loads(result.pop("metadata_json")); return result


def update_run(run_id, status=None, error=None, started_at=None, finished_at=None):
    if get_run(run_id) is None: raise ValueError(f"Run does not exist: {run_id}")
    if status is not None: status = _validate(status, RUN_STATUSES, "run status")
    connection = _connect(); connection.execute("UPDATE runs SET status = COALESCE(?, status), error = COALESCE(?, error), started_at = COALESCE(?, started_at), finished_at = COALESCE(?, finished_at), updated_at = ? WHERE id = ?", (status, error, started_at, finished_at, _utc_now(), run_id)); connection.commit(); connection.close(); return get_run(run_id)


def create_agent_session(run_id, provider, provider_session_id=None, status="STARTING", metadata=None):
    status = _validate(status, SESSION_STATUSES, "session status"); now, session_id = _utc_now(), _id(); connection = _connect()
    try:
        connection.execute("INSERT INTO agent_sessions (id, run_id, provider, provider_session_id, status, created_at, last_activity, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (session_id, run_id, provider, provider_session_id, status, now, now, json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True))); connection.commit()
    except sqlite3.IntegrityError as error:
        connection.rollback(); raise ValueError(f"Run does not exist: {run_id}") from error
    finally: connection.close()
    return get_agent_session(session_id)


def get_agent_session(session_id):
    connection = _connect(); row = connection.execute("SELECT * FROM agent_sessions WHERE id = ?", (session_id,)).fetchone(); connection.close()
    if not row: return None
    result = dict(row); result["metadata"] = json.loads(result.pop("metadata_json")); return result


def get_latest_provider_session_for_task(task_id, provider):
    connection = _connect(); row = connection.execute("""SELECT s.* FROM agent_sessions s JOIN runs r ON r.id = s.run_id WHERE r.task_id = ? AND s.provider = ? AND s.provider_session_id IS NOT NULL AND s.status IN ('ACTIVE','WAITING','COMPLETED') ORDER BY s.last_activity DESC LIMIT 1""", (task_id, provider)).fetchone(); connection.close()
    if not row: return None
    result = dict(row); result["metadata"] = json.loads(result.pop("metadata_json")); return result


def update_agent_session(session_id, status=None, provider_session_id=None, metadata=None):
    if get_agent_session(session_id) is None: raise ValueError(f"Agent session does not exist: {session_id}")
    if status is not None: status = _validate(status, SESSION_STATUSES, "session status")
    metadata_json = None if metadata is None else json.dumps(metadata, ensure_ascii=False, sort_keys=True); connection = _connect(); connection.execute("UPDATE agent_sessions SET status = COALESCE(?, status), provider_session_id = COALESCE(?, provider_session_id), metadata_json = COALESCE(?, metadata_json), last_activity = ? WHERE id = ?", (status, provider_session_id, metadata_json, _utc_now(), session_id)); connection.commit(); connection.close(); return get_agent_session(session_id)


def append_event(run_id, event_type, payload=None, session_id=None):
    event_type = str(event_type).upper()
    if event_type not in EVENT_TYPES: raise ValueError(f"Unsupported event type: {event_type}")
    connection = _connect(); sequence = int(connection.execute("SELECT COALESCE(MAX(sequence), 0) FROM events WHERE run_id = ?", (run_id,)).fetchone()[0]) + 1; event_id, now = _id(), _utc_now(); connection.execute("INSERT INTO events (id, run_id, session_id, type, sequence, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (event_id, run_id, session_id, event_type, sequence, json.dumps(payload or {}, ensure_ascii=False, sort_keys=True), now)); connection.commit(); connection.close(); return get_event(event_id)


def get_event(event_id):
    connection = _connect(); row = connection.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone(); connection.close()
    if not row: return None
    result = dict(row); result["payload"] = json.loads(result.pop("payload_json")); return result


def list_events(run_id):
    connection = _connect(); rows = connection.execute("SELECT * FROM events WHERE run_id = ? ORDER BY sequence", (run_id,)).fetchall(); connection.close(); results=[]
    for row in rows:
        item=dict(row); item["payload"]=json.loads(item.pop("payload_json")); results.append(item)
    return results


def create_artifact(project_id, artifact_type, name, path=None, run_id=None, content_hash=None, metadata=None):
    artifact_type = _validate(artifact_type, ARTIFACT_TYPES, "artifact type"); artifact_id, now = _id(), _utc_now(); connection = _connect(); connection.execute("INSERT INTO artifacts (id, project_id, run_id, type, name, path, content_hash, metadata_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (artifact_id, project_id, run_id, artifact_type, name, path, content_hash, json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True), now, now)); connection.commit(); connection.close(); return get_artifact(artifact_id)


def get_artifact(artifact_id):
    connection = _connect(); row = connection.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone(); connection.close()
    if not row: return None
    result=dict(row); result["metadata"]=json.loads(result.pop("metadata_json")); return result


def create_approval(run_id, artifact_id=None, content_hash=None, status="PENDING"):
    status = _validate(status, APPROVAL_STATUSES, "approval status"); approval_id, now = _id(), _utc_now(); connection = _connect(); connection.execute("INSERT INTO approvals (id, run_id, artifact_id, status, content_hash, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (approval_id, run_id, artifact_id, status, content_hash, now, now)); connection.commit(); connection.close(); return get_approval(approval_id)


def get_approval(approval_id):
    connection = _connect(); row = connection.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone(); connection.close(); return dict(row) if row else None


initialize_work_schema()
