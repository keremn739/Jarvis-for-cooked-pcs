import json

from llm import ask_router
from privacy import find_private_content


VALID_TYPES = {
    "MEMORY",
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


def validate_step(step, message):

    if not isinstance(step, dict):
        return None

    step_type = str(step.get("type", "")).upper()
    action = step.get("action")
    target = step.get("target")
    content = step.get("content")
    reason = step.get("reason")

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

        if action == "GET_TIME":

            if target is not None:
                return None

        elif action == "OPEN_APP":

            if not target:
                return None

            if target.lower() not in VALID_APPS:
                return None
            target = target.lower()

    # -------------------------------------------------
    # Non-tool steps cannot contain tool actions
    # -------------------------------------------------

    if step_type in {"MEMORY", "LOCAL", "CLOUD"}:

        if action is not None:
            return None

        if target is not None:
            return None

    # -------------------------------------------------
    # BLOCKED validation
    # -------------------------------------------------

    if step_type == "BLOCKED":

        if action is not None:
            return None

        if target is not None:
            return None

    # -------------------------------------------------
    # Step-level privacy barrier
    # -------------------------------------------------

    if step_type in {"MEMORY", "LOCAL", "CLOUD"} and content:

        private_findings = find_private_content(content)

        if private_findings:
            return {
                "type": "BLOCKED",
                "action": None,
                "target": None,
                "content": None,
                "reason": "Sensitive information request."
            }

    # -------------------------------------------------
    # Return validated step
    # -------------------------------------------------

    return {
        "type": step_type,
        "action": action,
        "target": target,
        "content": content,
        "reason": reason
    }


def route_message(message):

    try:

        raw_result = ask_router(message)

        plan = json.loads(raw_result)

    except (json.JSONDecodeError, TypeError, ValueError):

        return {
            "steps": [
                {
                    "type": "LOCAL",
                    "action": None,
                    "target": None,
                    "content": message,
                    "reason": "Invalid router output."
                }
            ]
        }

    if not isinstance(plan, dict):

        return {
            "steps": [
                {
                    "type": "LOCAL",
                    "action": None,
                    "target": None,
                    "content": message,
                    "reason": "Invalid router output."
                }
            ]
        }

    raw_steps = plan.get("steps")

    if not isinstance(raw_steps, list) or not raw_steps:

        return {
            "steps": [
                {
                    "type": "LOCAL",
                    "action": None,
                    "target": None,
                    "content": message,
                    "reason": "No valid plan returned."
                }
            ]
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

        has_blocked_step = any(
            step["type"] == "BLOCKED"
            for step in steps
        )

        if not has_blocked_step:

            steps = [
                {
                    "type": "BLOCKED",
                    "action": None,
                    "target": None,
                    "content": None,
                    "reason": "Sensitive information request."
                }
            ]

            steps.extend(
                step
                for step in []
            )

    # -------------------------------------------------
    # Final fallback
    # -------------------------------------------------

    if not steps:

        steps.append({
            "type": "LOCAL",
            "action": None,
            "target": None,
            "content": message,
            "reason": "No valid steps."
        })

    return {
        "steps": steps
    }