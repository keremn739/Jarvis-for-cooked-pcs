"""Versioned work specifications and approval integrity.

An approval is bound to the exact SHA-256 content hash of an artifact. Editing
the artifact therefore cannot silently preserve an old approval.
"""

from __future__ import annotations

import hashlib

from . import store


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def create_spec(project_id, name, content, *, run_id=None, artifact_type="SPEC"):
    digest = content_hash(content)
    return store.create_artifact(
        project_id,
        artifact_type,
        name,
        run_id=run_id,
        content_hash=digest,
        metadata={"content": content},
    )


def approve_spec(run_id, artifact_id):
    artifact = store.get_artifact(artifact_id)
    if artifact is None:
        raise ValueError(f"Artifact does not exist: {artifact_id}")
    if artifact["type"] not in {"SPEC", "PLAN"}:
        raise ValueError("Only SPEC and PLAN artifacts can be approved")
    return store.create_approval(run_id, artifact_id, artifact["content_hash"], "APPROVED")


def approval_is_current(approval_id):
    approval = store.get_approval(approval_id)
    if approval is None:
        return False
    if approval["status"] != "APPROVED":
        return False
    artifact = store.get_artifact(approval["artifact_id"])
    if artifact is None:
        return False
    return approval["content_hash"] == artifact["content_hash"]
