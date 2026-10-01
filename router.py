"""JARVIS request routing.

This module intentionally has no dependency on memory, privacy, or database
code. Local mode is managed by :mod:`mode`; online requests can produce TOOL,
CLOUD, or high-confidence WORK steps.
"""

import re

from api.groq_router import route_with_groq
from tools import TOOL_REGISTRY


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
    return _step("TOOL", None, action, target)


def _work_step(content):
    return _step("WORK", content)


def match_work(message):
    """Recognize explicit software/build work requests.

    This is intentionally conservative. Ambiguous requests continue through
    the normal CHAT/CLOUD path until the future Groq router can classify them.
    """
    text = message.lower().strip()
    software_evidence = (
        r"\b(?:software|code|coding|program(?:ming)?|script|python|javascript|"
        r"typescript|java|rust|go|project|repo|repository|app|application|"
        r"feature|function|class|file|bug|test|tests|test suite|website|"
        r"web app|api|database|module|package|build system)\b"
    )
    action = (
        r"\b(?:implement|build|develop|refactor|debug|fix|modify|add|remove|"
        r"create|write|change|update|run|execute)\b"
    )
    if not re.search(action, text) or not re.search(software_evidence, text):
        return None
    # Keep simple informational questions out of WORK.
    if re.match(r"^(?:how|what|why|can you explain)\b", text):
        return None
    return _work_step(message)


def _is_informational_open_app_request(message):
    text = message.lower().strip()
    # "How do I open Chrome?" asks for instructions. Do not use a blanket
    # "how" check: "How is Chrome performing? Open it" contains an action.
    return bool(
        re.match(
            r"^(?:how\s+(?:do|can|could|would)\s+i|tell\s+me\s+how\s+to|nasıl)\b",
            text,
        )
    )


def match_tool(message):
    """Recognize high-confidence, explicitly requested supported actions."""
    text = message.lower().strip()
    if re.search(
        r"\b(?:what time is it|current time|tell me the time|time right now)\b",
        text,
    ) or re.search(r"(?:saat kaç|şu an saat kaç)", text):
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
    parts = re.split(
        r"\s*,\s*then\s+|\s+then\s+|\s+and\s+",
        message,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
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
        if tool_step:
            steps.append(_step("TOOL", None, tool_step["action"], tool_step["target"]))
        else:
            steps.append(_step("CLOUD", part))
    return steps


def _is_original_intent_excerpt(content, message):
    """Reject WORK/CLOUD content containing text absent from the user request."""
    content_tokens = re.findall(r"\w+", content.casefold())
    message_tokens = re.findall(r"\w+", message.casefold())
    if not content_tokens or len(content_tokens) > len(message_tokens):
        return False
    width = len(content_tokens)
    return any(
        message_tokens[index : index + width] == content_tokens
        for index in range(len(message_tokens) - width + 1)
    )


def _validate_online_step(step, *, fallback_content=None):
    if not isinstance(step, dict):
        return None
    step_type = str(step.get("type", "")).strip().upper()
    if step_type == "TOOL":
        legacy_keys = {"type", "action", "target", "content"}
        schema_keys = legacy_keys | {"memory", "reason"}
        if set(step) not in (legacy_keys, schema_keys):
            return None
        if "memory" in step and step["memory"] is not False:
            return None
        if "reason" in step and step["reason"] is not None and not isinstance(step["reason"], str):
            return None
        content = step.get("content")
        if content is not None:
            return None
        action = str(step.get("action", "")).strip().upper()
        target = step.get("target")
        definition = TOOL_REGISTRY.get(action)
        if definition is None:
            return None
        if not definition["accepts_target"]:
            if target is not None:
                return None
            normalized_target = None
        else:
            if not isinstance(target, str) or target.lower() not in definition["targets"]:
                return None
            normalized_target = target.lower()
        return _tool_step(action, normalized_target)
    if step_type in {"WORK", "CLOUD"}:
        if set(step) != {"type", "action", "target", "content"}:
            return None
        content = step.get("content")
        if content is not None and not isinstance(content, str):
            return None
        if step.get("action") is not None or step.get("target") is not None:
            return None
        if (
            isinstance(content, str)
            and content.strip()
            and fallback_content
            and _is_original_intent_excerpt(content, fallback_content)
        ):
            if step_type == "WORK":
                return _work_step(content)
            return _step("CLOUD", content)
    return None


def route_online(message):
    """Route online requests to ordered TOOL/CLOUD/WORK steps."""
    if requested_mode(message) == "LOCAL":
        return {"steps": [], "switch_to": "LOCAL"}

    try:
        plan = route_with_groq(message)
        raw_steps = plan.get("steps") if isinstance(plan, dict) else None
        if (
            isinstance(plan, dict)
            and set(plan) == {"steps"}
            and isinstance(raw_steps, list)
            and raw_steps
        ):
            steps = [
                _validate_online_step(step, fallback_content=message)
                for step in raw_steps
            ]
            if all(step is not None for step in steps):
                return {"steps": steps}
    except Exception:
        pass

    return {"steps": [_step("CLOUD", message)]}


def route_local(message):
    """Route LOCAL mode to MEMORY, TOOL, WORK, or local Gemma FALLBACK."""
    if requested_mode(message) == "ONLINE":
        return {"steps": [], "switch_to": "ONLINE"}

    work_step = match_work(message)
    if work_step:
        return {"steps": [work_step]}

    tool_step = match_tool(message)
    if tool_step:
        return {"steps": [tool_step]}

    text = message.lower()
    asks_about_user = bool(
        re.search(
            r"\b(my|me|i have|i use|i am|i'm|what do you remember|what do i|what is my)\b|"
            r"(benim|bana|ben hangi|ekran kartım|hangi bilgisayar|ne hatırlıyorsun)",
            text,
        )
    )
    route_type = "MEMORY" if asks_about_user else "FALLBACK"
    return {"steps": [{"type": route_type, "content": message}]}
