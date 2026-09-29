"""CPU-local binary classifier for deciding whether a LOCAL request needs memory."""

import torch
import laya


# Keep Laya isolated from Ollama/Gemma and force CPU execution.
torch.set_num_interop_threads(1)
torch.set_num_threads(6)

_agent = None

_QUESTIONS = {
    "memory": {
        "type": "choice",
        "instructions": (
            "Does this request require Jarvis to retrieve or use "
            "personal information previously stored about the user?"
        ),
        "criteria": {
            "MEMORY": (
                "The request asks for, depends on, or requires personal "
                "information about the user stored in Jarvis memory."
            ),
            "NO_MEMORY": (
                "The request can be answered without using personal "
                "information stored about the user."
            ),
        },
    }
}


def _get_agent():
    global _agent

    if _agent is None:
        _agent = laya.load(
            "convaiinnovations/laya",
            device="cpu",
        )

    return _agent


def needs_memory(message):
    """Return True when Laya classifies the LOCAL request as memory-dependent."""

    result = _get_agent().predict(message, _QUESTIONS)

    choice = result["answers"]["memory"]["choice"]

    if choice == "MEMORY":
        return True

    if choice == "NO_MEMORY":
        return False

    # Conservative fallback: if Laya produces an unexpected label,
    # do not assume that memory is required.
    return False
