"""Bridge durable JARVIS work state to an agent provider.

This is the application boundary between the WorkManager and provider adapters.
It translates provider protocol events into durable JARVIS events and keeps
provider-specific protocol objects out of the rest of the application.
"""

from __future__ import annotations

from typing import Callable, Optional

from .manager import WorkManager
from .providers.codex import CodexProvider


class WorkRunner:
    def __init__(
        self,
        manager: Optional[WorkManager] = None,
        provider_factory: Callable[..., CodexProvider] = CodexProvider,
    ):
        self.manager = manager or WorkManager()
        self.provider_factory = provider_factory

    def run_codex_turn(self, run_id: str, message: str, *, cwd=None):
        """Execute one Codex turn for an existing Run and persist its events."""
        run = self.manager.get_run(run_id)
        if run is None:
            raise ValueError(f"Run does not exist: {run_id}")

        session = self.manager.create_session(run_id, "codex")
        provider = self.provider_factory(cwd=cwd, on_notification=lambda event: self._record_provider_event(run_id, session["id"], event))

        try:
            provider.start()
            thread = provider.create_thread(cwd=cwd)
            self.manager.attach_provider_session(session["id"], thread["id"])
            self.manager.start_run(run_id)

            for notification in provider.send(message, cwd=cwd):
                self._record_provider_event(run_id, session["id"], notification)

            self.manager.complete_run(run_id)
            return self.manager.get_run(run_id)
        except Exception as error:
            self.manager.fail_run(run_id, error)
            raise
        finally:
            provider.stop()

    def _record_provider_event(self, run_id, session_id, notification):
        method = notification.get("method") if isinstance(notification, dict) else None
        if not method:
            return

        event_type = self._event_type(method)
        payload = notification.get("params", {})
        if event_type is None:
            # Preserve useful but currently unmapped protocol events without
            # pretending that JARVIS understands their semantics yet.
            return
        self.manager.record_event(run_id, event_type, payload, session_id)

    @staticmethod
    def _event_type(method):
        mapping = {
            "turn/started": "TURN_STARTED",
            "item/agentMessage/delta": "AGENT_MESSAGE_DELTA",
            "item/agentMessage/completed": "AGENT_MESSAGE_COMPLETED",
            "item/commandExecution/started": "COMMAND_STARTED",
            "item/commandExecution/completed": "COMMAND_COMPLETED",
            "item/toolCall/started": "TOOL_STARTED",
            "item/toolCall/completed": "TOOL_COMPLETED",
            "turn/completed": "TURN_COMPLETED",
        }
        return mapping.get(method)
