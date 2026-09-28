import json

from llm import ask_router
from privacy import find_private_content


VALID_TYPES = {
    "LOCAL",
    "CLOUD",
    "TOOL",
    "BLOCKED"
}

VALID_TOOLS = {
    "GET_TIME",
    "OPEN_APP"
}

VALID_APPS = {
    "chrome",
    "notepad",
    "calculator"
}


def is_instructional_open_app_request(message):
    text = message.lower().strip()

    instructional_markers = (
        "how ",
        "how do ",
        "how can ",
        "tell me how ",
        "nasıl ",
    )

    return any(marker in text for marker in instructional_markers)


def validate_step(step, message):

    if not isinstance(step, dict):
        return None

    raw_step_type = str(step.get("type", "")).strip().upper()

    # Gemma sometimes combines the LOCAL route and memory flag into the
    # type string even though the schema defines memory as a separate field.
    # Normalize that specific formatting error instead of rejecting an
    # otherwise correct personal-memory decision.
    if raw_step_type in {
        "LOCAL + MEMORY=TRUE",
        "LOCAL+MEMORY=TRUE",
    }:
        step_type = "LOCAL"
        step["memory"] = True
    else:
        step_type = raw_step_type

    action = step.get("action")
    target = step.get("target")
    content = step.get("content")
    reason = step.get("reason")
    memory = bool(step.get("memory", False))

    if step_type not in VALID_TYPES:
        return None

    if action is not None:
        action = str(action).upper()

    if target is not None:
        target = str(target)

    if content is not None:
        content = str(content)

    if reason is not None:
        reason = str(reason)

    # -------------------------------------------------
    # TOOL validation
    # -------------------------------------------------

    if step_type == "TOOL":

        if action not in VALID_TOOLS:
            return None

        if memory:
            return None

        if action == "GET_TIME":

            if target is not None:
                return None

        elif action == "OPEN_APP":

            if not target:
                return None

            if target.lower() not in VALID_APPS:
                return None

            if is_instructional_open_app_request(message):
                return {
                    "type": "LOCAL",
                    "action": None,
                    "target": None,
                    "content": message,
                    "memory": False,
                    "reason": "Instructional request, not an execution request."
                }

            target = target.lower()

    # -------------------------------------------------
    # LOCAL validation
    # -------------------------------------------------

    if step_type == "LOCAL":

        if action is not None:
            return None

        if target is not None:
            return None

        if not content:
            return None

        # LOCAL is the only route allowed to use memory.
        if memory:
            memory = True

    # -------------------------------------------------
    # CLOUD validation
    # -------------------------------------------------

    if step_type == "CLOUD":

        if action is not None:
            return None

        if target is not None:
            return None

        if not content:
            return None

        # Critical privacy invariant:
        #
        # CLOUD + memory=True is never allowed.
        #
        # If Gemma accidentally combines them, force the request
        # back onto the local path.
        if memory:
            return {
                "type": "LOCAL",
                "action": None,
                "target": None,
                "content": content,
                "memory": True,
                "reason": "Personal memory request kept local."
            }

    # -------------------------------------------------
    # BLOCKED validation
    # -------------------------------------------------

    if step_type == "BLOCKED":

        if action is not None:
            return None

        if target is not None:
            return None

        if memory:
            return None

    # -------------------------------------------------
    # Privacy barrier
    # -------------------------------------------------

    if step_type in {"LOCAL", "CLOUD"} and content:

        private_findings = find_private_content(content)

        if private_findings:

            return {
                "type": "BLOCKED",
                "action": None,
                "target": None,
                "content": None,
                "memory": False,
                "reason": "Sensitive information request."
            }

    return {
        "type": step_type,
        "action": action,
        "target": target,
        "content": content,
        "memory": memory,
        "reason": reason
    }


def route_message(message):

    try:

        raw_result = ask_router(message)

        plan = json.loads(raw_result)

    except (json.JSONDecodeError, TypeError, ValueError):

        return {
            "steps": [{
                "type": "LOCAL",
                "action": None,
                "target": None,
                "content": message,
                "memory": False,
                "reason": "Invalid router output."
            }]
        }

    if not isinstance(plan, dict):

        return {
            "steps": [{
                "type": "LOCAL",
                "action": None,
                "target": None,
                "content": message,
                "memory": False,
                "reason": "Invalid router output."
            }]
        }

    raw_steps = plan.get("steps")

    if not isinstance(raw_steps, list) or not raw_steps:

        return {
            "steps": [{
                "type": "LOCAL",
                "action": None,
                "target": None,
                "content": message,
                "memory": False,
                "reason": "No valid plan returned."
            }]
        }

    steps = []

    for step in raw_steps:

        validated_step = validate_step(step, message)

        if validated_step:
            steps.append(validated_step)

    # -------------------------------------------------
    # Raw-message privacy safety net
    # -------------------------------------------------

    private_findings = find_private_content(message)

    if private_findings:

        steps = [{
            "type": "BLOCKED",
            "action": None,
            "target": None,
            "content": None,
            "memory": False,
            "reason": "Sensitive information request."
        }]

    # -------------------------------------------------
    # Final fallback
    # -------------------------------------------------

    if not steps:

        steps.append({
            "type": "LOCAL",
            "action": None,
            "target": None,
            "content": message,
            "memory": False,
            "reason": "No valid steps."
        })

    return {
        "steps": steps
    }
