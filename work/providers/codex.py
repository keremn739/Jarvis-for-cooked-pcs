"""Codex app-server adapter.

JARVIS talks to Codex through its documented JSON-RPC app-server interface rather
than scraping terminal output. The adapter owns transport/protocol details;
WorkManager owns durable application state.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional


class CodexProviderError(RuntimeError):
    """Raised when the Codex app-server cannot be used."""


class CodexProvider:
    """One long-lived Codex app-server process for one JARVIS agent session."""

    def __init__(
        self,
        *,
        command: Optional[list[str]] = None,
        cwd: Optional[str | Path] = None,
        env: Optional[dict[str, str]] = None,
        on_notification: Optional[Callable[[dict], None]] = None,
    ):
        self.command = command or ["codex", "app-server", "--listen", "stdio://"]
        self.cwd = str(cwd) if cwd is not None else None
        self.env = {**os.environ, **(env or {})}
        self.on_notification = on_notification

        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._messages: queue.Queue[dict] = queue.Queue()
        self._pending: dict[str | int, queue.Queue[dict]] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._request_id = 0
        self._thread_id: str | None = None
        self._turn_id: str | None = None
        self._initialized = False
        self._stopped = False

    @property
    def thread_id(self):
        return self._thread_id

    @property
    def turn_id(self):
        return self._turn_id

    @property
    def running(self):
        return self._process is not None and self._process.poll() is None

    def start(self):
        if self.running:
            return
        self._stopped = False
        self._process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name="jarvis-codex-reader",
            daemon=True,
        )
        self._reader_thread.start()

        self._request(
            "initialize",
            {
                "clientInfo": {
                    "name": "jarvis",
                    "title": "JARVIS",
                    "version": "0.1.0",
                }
            },
            timeout=30,
        )
        self._notify("initialized", {})
        self._initialized = True

    def create_thread(self, *, cwd: Optional[str | Path] = None, ephemeral=False):
        self._require_started()
        params = {"ephemeral": ephemeral}
        selected_cwd = str(cwd) if cwd is not None else self.cwd
        if selected_cwd:
            params["cwd"] = selected_cwd
        result = self._request("thread/start", params, timeout=60)
        try:
            self._thread_id = result["thread"]["id"]
        except (KeyError, TypeError) as exc:
            raise CodexProviderError(f"Codex returned no thread id: {result!r}") from exc
        return result["thread"]

    def resume_thread(self, thread_id: str):
        self._require_started()
        result = self._request("thread/resume", {"threadId": thread_id}, timeout=60)
        self._thread_id = thread_id
        return result.get("thread", result)

    def send(self, message: str, *, cwd: Optional[str | Path] = None):
        """Run one turn and yield protocol notifications until completion."""
        self._require_started()
        if not self._thread_id:
            self.create_thread(cwd=cwd)

        params = {
            "threadId": self._thread_id,
            "input": [{"type": "text", "text": message}],
        }
        selected_cwd = str(cwd) if cwd is not None else None
        if selected_cwd:
            params["cwd"] = selected_cwd

        request_id = self._next_request_id()
        response_queue: queue.Queue[dict] = queue.Queue()
        self._register_pending(request_id, response_queue)
        try:
            self._write({"jsonrpc": "2.0", "id": request_id, "method": "turn/start", "params": params})
            response = response_queue.get(timeout=120)
            if "error" in response:
                raise CodexProviderError(response["error"])
            self._turn_id = response.get("result", {}).get("turn", {}).get("id")

            while True:
                notification = self._messages.get(timeout=600)
                if notification.get("method") == "turn/completed":
                    turn = notification.get("params", {}).get("turn", {})
                    if self._turn_id is None or turn.get("id") == self._turn_id:
                        yield notification
                        status = turn.get("status")
                        if status != "completed":
                            raise CodexProviderError(
                                f"Codex turn ended with status {status!r}: {turn.get('error')}"
                            )
                        return
                else:
                    yield notification
        finally:
            self._unregister_pending(request_id)

    def interrupt(self):
        if not self._thread_id or not self._turn_id:
            return
        self._notify(
            "turn/interrupt",
            {"threadId": self._thread_id, "turnId": self._turn_id},
        )

    def respond(self, request_id: str | int, result: dict):
        """Answer a server-initiated JSON-RPC request, such as an approval."""
        self._write({"jsonrpc": "2.0", "id": request_id, "result": result})

    def stop(self):
        self._stopped = True
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    close = stop

    def _read_loop(self):
        assert self._process is not None
        stdout = self._process.stdout
        assert stdout is not None
        try:
            for line in stdout:
                if self._stopped:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "id" in message and ("result" in message or "error" in message):
                    request_id = message["id"]
                    with self._pending_lock:
                        waiter = self._pending.get(request_id)
                    if waiter is not None:
                        waiter.put(message)
                        continue

                if "id" in message and "method" in message:
                    # Server-initiated requests, e.g. command/file approvals.
                    self._messages.put(message)
                    if self.on_notification:
                        self.on_notification(message)
                    continue

                self._messages.put(message)
                if self.on_notification:
                    self.on_notification(message)
        finally:
            if not self._stopped:
                self._messages.put({
                    "method": "codex/processExited",
                    "params": {"returncode": self._process.poll()},
                })

    def _request(self, method: str, params: dict, *, timeout: float):
        request_id = self._next_request_id()
        waiter: queue.Queue[dict] = queue.Queue()
        self._register_pending(request_id, waiter)
        try:
            self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            try:
                response = waiter.get(timeout=timeout)
            except queue.Empty as exc:
                raise CodexProviderError(f"Timed out waiting for Codex: {method}") from exc
            if "error" in response:
                raise CodexProviderError(response["error"])
            return response.get("result", {})
        finally:
            self._unregister_pending(request_id)

    def _notify(self, method: str, params: dict):
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _write(self, message: dict):
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise CodexProviderError("Codex app-server is not running")
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            process.stdin.write(payload + "\n")
            process.stdin.flush()

    def _next_request_id(self):
        self._request_id += 1
        return self._request_id

    def _register_pending(self, request_id, waiter):
        with self._pending_lock:
            self._pending[request_id] = waiter

    def _unregister_pending(self, request_id):
        with self._pending_lock:
            self._pending.pop(request_id, None)

    def _require_started(self):
        if not self.running or not self._initialized:
            raise CodexProviderError("CodexProvider.start() must succeed first")
