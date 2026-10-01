"""Build the deterministic context envelope passed to an agent provider.

The context envelope is application state, not a transcript. It identifies the
project/task/run and the approved artifacts that govern the current execution.
"""

from __future__ import annotations

from . import store
from .specs import approval_is_current


def build_run_context(run_id):
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"Run does not exist: {run_id}")

    task = store.get_task(run["task_id"])
    project = store.get_project(task["project_id"])
    if task is None or project is None:
        raise ValueError("Run references missing task/project")

    context = {
        "project": {
            "id": project["id"],
            "name": project["name"],
            "description": project["description"],
            "root_path": project["root_path"],
        },
        "task": {
            "id": task["id"],
            "title": task["title"],
            "description": task["description"],
            "priority": task["priority"],
        },
        "run": {
            "id": run["id"],
            "status": run["status"],
        },
        "approved_artifacts": [],
    }

    # Approval lookup is intentionally explicit rather than hidden in an LLM
    # prompt. This lets the runner refuse stale approvals deterministically.
    connection = store._connect()
    rows = connection.execute(
        "SELECT id, artifact_id, content_hash, status FROM approvals WHERE run_id = ? ORDER BY created_at",
        (run_id,),
    ).fetchall()
    connection.close()

    for row in rows:
        approval = dict(row)
        if approval["status"] != "APPROVED" or not approval_is_current(approval["id"]):
            continue
        artifact = store.get_artifact(approval["artifact_id"])
        if artifact:
            context["approved_artifacts"].append({
                "id": artifact["id"],
                "type": artifact["type"],
                "name": artifact["name"],
                "content_hash": artifact["content_hash"],
                "content": artifact["metadata"].get("content"),
            })

    return context


def render_provider_context(context):
    """Render only deterministic work context; conversational history stays with the provider."""
    lines = [
        f"Project: {context['project']['name']}",
        f"Project root: {context['project']['root_path'] or '(not set)'}",
        f"Task: {context['task']['title']}",
        f"Task description: {context['task']['description']}",
        f"Run ID: {context['run']['id']}",
    ]
    if context["approved_artifacts"]:
        lines.append("Approved artifacts:")
        for artifact in context["approved_artifacts"]:
            lines.append(f"- {artifact['type']}: {artifact['name']} (sha256={artifact['content_hash']})")
            if artifact.get("content"):
                lines.append(artifact["content"])
    return "\\n".join(lines)
