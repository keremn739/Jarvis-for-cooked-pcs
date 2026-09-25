"""Privacy-filtered interaction history, separate from semantic memory."""

import queue
import re
import threading

from database import add_interaction
from privacy import find_private_content


def redact_sensitive_content(message):
    """Replace detected sensitive spans so their values are never persisted."""
    findings = find_private_content(message)
    for finding in sorted(findings, key=lambda item: item["start"], reverse=True):
        message = (
            message[:finding["start"]]
            + "[REDACTED]"
            + message[finding["end"]:]
        )

    # The existing privacy patterns include category words (for example,
    # "password") rather than the credentials that may follow them. Avoid
    # persisting a credential value attached to a detected credential label.
    credential_pattern = re.compile(
        r"(?i)(\b(?:password\w*|passcode\w*|şifre|api\s+key\w*|"
        r"api\s+anahtarı|token\w*|secret\w*|private\s+key)\b"
        r"\s*(?:is|are|=|:|to)?\s*)([^\s,;]+)"
    )
    message = credential_pattern.sub(r"\1[REDACTED]", message)
    redacted_label_pattern = re.compile(
        r"(?i)(\[REDACTED\]\s*(?:is|are|=|:|to)\s*)([^\s,;]+)"
    )
    return redacted_label_pattern.sub(r"\1[REDACTED]", message)


def record_interaction(message):
    """Store a single raw user message after removing sensitive content."""
    return add_interaction(redact_sensitive_content(str(message)))


class InteractionHistoryWriter:
    """Queue writes for a daemon worker so SQLite is off the response path."""

    _STOP = object()

    def __init__(self, write=record_interaction):
        self._queue = queue.Queue()
        self._write = write
        self._closed = False
        self._close_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, message):
        with self._close_lock:
            if not self._closed:
                self._queue.put_nowait(message)

    def flush(self):
        """Wait until all currently queued writes have finished."""
        self._queue.join()

    def close(self):
        """Drain queued writes and stop the worker during application exit."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put_nowait(self._STOP)
        self._queue.join()
        self._thread.join()

    def _run(self):
        while True:
            message = self._queue.get()
            try:
                if message is self._STOP:
                    return
                self._write(message)
            except Exception:
                # History is best-effort and must never affect the assistant.
                pass
            finally:
                self._queue.task_done()


_writer = InteractionHistoryWriter()


def enqueue_interaction(message):
    """Schedule a completed interaction for best-effort persistence."""
    _writer.enqueue(message)


def flush_interactions():
    """Wait for pending history writes without stopping the worker."""
    _writer.flush()


def close_interaction_history():
    """Drain and stop the history worker before process shutdown."""
    _writer.close()
