"""Entry point for routing a WORK request into a durable Codex run."""

from pathlib import Path

from . import store
from .manager import WorkManager
from .runner import WorkRunner


def dispatch_work(message, *, project_root=None, manager=None, runner=None):
    """Reuse the project's active task so successive WORK turns share context."""
    manager = manager or WorkManager()
    root = Path(project_root or Path.cwd()).expanduser().resolve()
    project = store.get_project_by_root(str(root))
    if project is None:
        project = manager.create_project("JARVIS", "JARVIS source project", str(root))

    task = store.get_latest_active_task(project["id"])
    if task is None:
        task = manager.create_task(project["id"], message, message)

    run = manager.create_run(task["id"], {"source": "router", "provider": "codex"})
    runner = runner or WorkRunner(manager)
    return runner.run_codex_turn(run["id"], message, cwd=str(root))
