"""Online request routing.

This module intentionally has no dependency on memory, privacy, or database
code. Local mode is managed by :mod:`mode`; online requests can only produce
TOOL and CLOUD steps.
"""

import json
import re

from llm import ask_router


VALID_TOOLS = {"GET_TIME", "OPEN_APP"}
VALID_APPS = {"chrome", "notepad", "calculator"}

LOCAL_MODE_COMMANDS = (
    "go into local mode",
    "enter local mode",
    "switch to local mode",
    "activate local mode",
    "yerel moda geç",
    "yerel moda gir",
)

ONLINE_MODE_COMMANDS = (
    "go into online mode",
    "return to online mode",
    "switch to online mode",
    "back to online mode",
    "çevrimiçi moda geç",
    "normal moda dön",
)


def _normalized_command(message):
    text = str(message).strip().lower()
    text = re.sub(r"^[\s,;.!?]*(?:jarvis\s*[, :]\s*)?", "", text)
    return re.sub(r"[\s.!?]+$", "", text)


def requested_mode(message):
    """Return an explicitly requested mode, or ``None``."""
    text = _normalized_command(message)
    if text in LOCAL_MODE_COMMANDS:
        return "LOCAL"
    if text in ONLINE_MODE_COMMANDS:
        return "ONLINE"
    return None


def _step(step_type, content, action=None, target=None):
    return {
        "type": step_type,
        "action": action,
        "target": target,
        "content": content,
        "memory": False,
        "reason": None,
    }


def _tool_step(action, target=None):
    return {
        "type": "TOOL",
        "action": action,
        "target": target,
        "content": None,
        "memory": False,
        "reason": None,
    }


def _is_informational_open_app_request(message):
    text = message.lower().strip()
    # "How do I open Chrome?" asks for instructions. Do not use a blanket
    # "how" check: "How is Chrome performing? Open it" contains an action.
    return bool(re.match(r"^(?:how\s+(?:do|can|could|would)\s+i|tell\s+me\s+how\s+to|nasıl)\b", text))


def match_tool(message):
    """Recognize high-confidence, explicitly requested supported actions."""
    text = message.lower().strip()
    if re.search(r"\b(?:what time is it|current time|tell me the time|time right now)\b", text) or \
            re.search(r"(?:saat kaç|şu an saat kaç)", text):
        return _tool_step("GET_TIME")

    if _is_informational_open_app_request(text):
        return None

    app_patterns = (
        ("chrome", r"(?:google\s+)?chrome"),
        ("notepad", r"notepad"),
        ("calculator", r"(?:the\s+)?calculator|hesap makinesi(?:ni)?"),
    )
    action_words = r"\b(?:open|launch|start|run)\b|(?:aç|başlat)"
    if not re.search(action_words, text):
        return None
    for app, pattern in app_patterns:
        if re.search(pattern, text):
            return _tool_step("OPEN_APP", app)
    return None


def _split_ordered_intents(message):
    """Split common two-part requests while preserving the written order."""
    parts = re.split(r"\s*,\s*then\s+|\s+then\s+|\s+and\s+", message, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    left, right = (part.strip(" ,.!?") for part in parts)
    if not left or not right:
        return None
    left_tool, right_tool = match_tool(left), match_tool(right)
    if bool(left_tool) == bool(right_tool):
        return None
    steps = []
    for part, tool_step in ((left, left_tool), (right, right_tool)):
        steps.append(tool_step if tool_step else _step("CLOUD", part))
    return steps


def _validate_online_step(step):
    if not isinstance(step, dict):
        return None
    step_type = str(step.get("type", "")).strip().upper()
    if step_type == "TOOL":
        action = str(step.get("action", "")).strip().upper()
        target = step.get("target")
        if action not in VALID_TOOLS or bool(step.get("memory", False)):
            return None
        if action == "GET_TIME":
            if target is not None:
                return None
        elif not target or str(target).lower() not in VALID_APPS:
            return None
        return _tool_step(action, str(target).lower() if target is not None else None)
    if step_type == "CLOUD" and not step.get("memory", False):
        content = step.get("content")
        if content:
            return _step("CLOUD", str(content))
    return None


def route_online(message):
    """Route online requests to ordered TOOL/CLOUD steps only."""
    if requested_mode(message) == "LOCAL":
        return {"steps": [], "switch_to": "LOCAL"}

    split_steps = _split_ordered_intents(message)
    if split_steps:
        return {"steps": split_steps}

    direct_tool = match_tool(message)
    if direct_tool:
        return {"steps": [direct_tool]}

    try:
        raw_result = ask_router(message)
        plan = json.loads(raw_result)
        raw_steps = plan.get("steps") if isinstance(plan, dict) else None
        if isinstance(raw_steps, list) and raw_steps:
            steps = [_validate_online_step(step) for step in raw_steps]
            steps = [step for step in steps if step]
            if steps:
                return {"steps": steps}
    except (json.JSONDecodeError, TypeError, ValueError, OSError, TimeoutError):
        pass

    # Default authority for ordinary questions belongs to the online mode.
    return {"steps": [_step("CLOUD", message)]}


def route_local(message):
    """Route LOCAL mode to MEMORY, TOOL, or local Gemma FALLBACK."""
    if requested_mode(message) == "ONLINE":
        return {"steps": [], "switch_to": "ONLINE"}

    tool_step = match_tool(message)
    if tool_step:
        return {"steps": [tool_step]}

    text = message.lower()
    asks_about_user = bool(re.search(
        r"\b(my|me|i have|i use|i am|i'm|what do you remember|what do i|what is my)\b|"
        r"(benim|bana|ben hangi|ekran kartım|hangi bilgisayar|ne hatırlıyorsun)",
        text,
    ))
    route_type = "MEMORY" if asks_about_user else "FALLBACK"
    return {"steps": [{"type": route_type, "content": message}]}
