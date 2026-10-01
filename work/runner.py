"""Bridge durable JARVIS work state to an agent provider."""

from __future__ import annotations

from typing import Callable, Optional

from . import store
from .context import build_run_context, render_provider_context
from .manager import WorkManager
from .providers.codex import CodexProvider


class WorkRunner:
    """Execute provider turns while keeping JARVIS state authoritative."""

    DEFAULT_ROTATION_TURNS = 8

    def __init__(
        self,
        manager: Optional[WorkManager] = None,
        provider_factory: Callable[..., CodexProvider] = CodexProvider,
        rotation_turns: int = DEFAULT_ROTATION_TURNS,
    ):
        self.manager = manager or WorkManager()
        self.provider_factory = provider_factory
        if rotation_turns < 1:
            raise ValueError("rotation_turns must be at least 1")
        self.rotation_turns = rotation_turns

    def run_codex_turn(self, run_id: str, message: str, *, cwd=None):
        run = self.manager.get_run(run_id)
        if run is None:
            raise ValueError(f"Run does not exist: {run_id}")
        task = store.get_task(run["task_id"])
        if task is None:
            raise ValueError(f"Task does not exist: {run['task_id']}")

        previous_session = store.get_latest_provider_session_for_task(task["id"], "codex")
        session = self.manager.create_session(run_id, "codex")
        provider = self.provider_factory(cwd=cwd)
        resumed = bool(previous_session and previous_session["provider_session_id"])

        try:
            provider.start()
            if resumed:
                provider.resume_thread(previous_session["provider_session_id"])
                self.manager.attach_provider_session(session["id"], previous_session["provider_session_id"])
            else:
                thread = provider.create_thread(cwd=cwd)
                self.manager.attach_provider_session(session["id"], thread["id"])

            self.manager.start_run(run_id)
            prompt = message
            if not resumed:
                context = build_run_context(run_id)
                prompt = render_provider_context(context) + "\\n\\nUser request:\\n" + message

            for notification in provider.send(prompt, cwd=cwd):
                self._record_provider_event(run_id, session["id"], notification)

            turns += 1
            self.manager.update_session_metadata(
                session["id"],
                {"turn_count": turns, "handover_summary": handover},
            )
            self.manager.complete_run(run_id)
            self.manager.transition_session(session["id"], "COMPLETED")
            return self.manager.get_run(run_id)
        except Exception as error:
            try:
                self.manager.fail_run(run_id, error)
                self.manager.transition_session(session["id"], "FAILED")
            finally:
                raise
        finally:
            provider.stop()

    def _create_handover(self, provider, run_id, session_id, project_id, cwd):
        request = (
            "Create a concise handover summary for another coding-agent session. "
            "Include current objective, work completed, files changed, important "
            "decisions, current implementation state, known issues, tests/results, "
            "and exact next steps. Do not invent anything. Return only the summary."
        )
        parts = []
        for notification in provider.send(request, cwd=cwd):
            self._record_provider_event(run_id, session_id, notification)
            text = self._extract_completed_text(notification)
            if text:
                parts.append(text)
        summary = "\n".join(parts).strip()
        if not summary:
            raise RuntimeError("Codex produced no handover summary")
        self.manager.create_handover_artifact(
            project_id, run_id, summary, {"source_session_id": session_id}
        )
        self.manager.record_event(
            run_id, "HANDOVER_CREATED",
            {"source_session_id": session_id, "summary": summary}, session_id
        )
        return summary

    @staticmethod
    def _extract_completed_text(notification):
        if notification.get("method") != "item/agentMessage/completed":
            return ""
        item = notification.get("params", {}).get("item", {})
        if isinstance(item.get("text"), str):
            return item["text"]
        content = item.get("content")
        if isinstance(content, list):
            return "\n".join(
                part["text"] for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        return ""

    def _record_provider_event(self, run_id, session_id, notification):
        method = notification.get("method") if isinstance(notification, dict) else None
        if not method:
            return
        event_type = self._event_type(method)
        if event_type is None:
            return
        self.manager.record_event(run_id, event_type, notification.get("params", {}), session_id)

    @staticmethod
    def _event_type(method):
        return {
            "turn/started": "TURN_STARTED",
            "item/agentMessage/delta": "AGENT_MESSAGE_DELTA",
            "item/agentMessage/completed": "AGENT_MESSAGE_COMPLETED",
            "item/commandExecution/started": "COMMAND_STARTED",
            "item/commandExecution/completed": "COMMAND_COMPLETED",
            "item/toolCall/started": "TOOL_STARTED",
            "item/toolCall/completed": "TOOL_COMPLETED",
            "turn/completed": "TURN_COMPLETED",
        }.get(method)
