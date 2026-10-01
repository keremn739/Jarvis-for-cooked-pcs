"""Artifact persistence helpers.

Artifacts are durable references to files/content produced or consumed by a
run. The database stores metadata and hashes; large files remain on disk.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import store


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def register_file(project_id, path, *, artifact_type="OUTPUT", run_id=None, name=None):
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise ValueError(f"Artifact file does not exist: {file_path}")
    data = file_path.read_bytes()
    return store.create_artifact(
        project_id,
        artifact_type,
        name or file_path.name,
        path=str(file_path),
        run_id=run_id,
        content_hash=sha256_bytes(data),
        metadata={"size": len(data)},
    )


def read_registered_artifact(artifact_id):
    artifact = store.get_artifact(artifact_id)
    if artifact is None:
        raise ValueError(f"Artifact does not exist: {artifact_id}")
    if not artifact["path"]:
        return artifact["metadata"].get("content")
    path = Path(artifact["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")
