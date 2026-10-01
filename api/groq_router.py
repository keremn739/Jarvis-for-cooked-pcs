"""Groq Structured Outputs client for Jarvis online routing."""

from __future__ import annotations

import json
import os

from tools import TOOL_REGISTRY


MODEL = "openai/gpt-oss-20b"
REASONING_EFFORT = "low"
MAX_COMPLETION_TOKENS = 512
TEMPERATURE = 0.0


def _tool_step_schema():
    actions = list(TOOL_REGISTRY)
    targets = list(
        dict.fromkeys(
            target
            for definition in TOOL_REGISTRY.values()
            for target in definition["targets"]
        )
    )
    return {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["TOOL"]},
            "action": {"type": "string", "enum": actions},
            "target": {"type": ["string", "null"], "enum": targets},
            "content": {"type": ["string", "null"]},
            "memory": {"type": "boolean"},
            "reason": {"type": ["string", "null"]},
        },
        "required": ["type", "action", "target", "content", "memory", "reason"],
        "additionalProperties": False,
    }


def _text_step_schema(step_type):
    return {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": [step_type]},
            "action": {"type": "null"},
            "target": {"type": "null"},
            "content": {"type": "string"},
        },
        "required": ["type", "action", "target", "content"],
        "additionalProperties": False,
    }


ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "anyOf": [
                    _tool_step_schema(),
                    _text_step_schema("WORK"),
                    _text_step_schema("CLOUD"),
                ]
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}


def _tool_instructions():
    lines = []
    for action, definition in TOOL_REGISTRY.items():
        targets = definition["targets"]
        target_text = "null" if targets == (None,) else ", ".join(targets)
        lines.append(
            f'- {action}: {definition["purpose"]}; trigger: '
            f'{definition["trigger"]}; target: {target_text}.'
        )
    return "\n".join(lines)


SYSTEM_PROMPT = f"""YOU ARE A ROUTER, NOT THE ASSISTANT. NEVER ANSWER THE USER.
Never invent facts/results.

TOOLS (use only for explicit supported actions):
{_tool_instructions()}

WORK means software/project execution handled by Jarvis Work Manager. This
includes running tests or the test suite, building a project, implementing a
feature, modifying/refactoring/debugging code, and creating/changing project
files. "Run the test suite." must produce a WORK step.

CLOUD means informational, explanatory, research, or general-answer requests.
"How do I open Chrome?", "Can you explain how to open Chrome?", and "Tell me
how to launch Chrome." = CLOUD.

"Open Chrome." = TOOL / OPEN_APP / chrome. TOOL means Jarvis performs the
supported action. Preserve user order and split each multi-intent request into
separate ordered steps. Never omit later intents. TOOL content is null.
WORK/CLOUD content represents the user's requested operation/question, never
an answer. Return only the strict structured routing plan."""


def route_with_groq(message: str, *, client=None) -> dict:
    """Return a Groq-generated online routing plan.

    ``client`` is injectable for tests; normal callers use ``GROQ_API_KEY``.
    API errors and malformed output propagate to the router's safe CLOUD
    fallback.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if client is None:
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        from groq import Groq

        client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "jarvis_online_route",
                "strict": True,
                "schema": ROUTER_SCHEMA,
            },
        },
        reasoning_effort=REASONING_EFFORT,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
    )
    content = response.choices[0].message.content
    plan = json.loads(content) if isinstance(content, str) else content
    if not isinstance(plan, dict):
        raise ValueError("Groq router response must be a JSON object")
    return plan
