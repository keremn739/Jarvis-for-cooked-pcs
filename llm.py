import json
import time
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"
LOCAL_MODEL = "gemma3:4b"
OLLAMA_TIMEOUT = 120


def _ollama_generate(prompt, *, stream=False, output_format=None):
    data = {
        "model": LOCAL_MODEL,
        "prompt": prompt,
        "stream": stream,
        "think": False,
    }
    if output_format is not None:
        data["format"] = output_format

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT)


def ask_llm(message):
    start = time.time()
    response = _ollama_generate(message, stream=True)
    answer = ""
    first_token_time = None

    while True:
        line = response.readline()
        if not line:
            break
        result = json.loads(line)
        chunk = result.get("response", "")
        if first_token_time is None and chunk:
            first_token_time = time.time()
        answer += chunk
        print(chunk, end="", flush=True)
        if result.get("done", False):
            break

    end = time.time()
    print()
    if first_token_time is None:
        print("[Time to first token: unavailable]")
    else:
        print(f"[Time to first token: {first_token_time - start:.2f} seconds]")
    print(f"[Total response time: {end - start:.2f} seconds]")
    return answer


def ask_llm_json(message):
    response = _ollama_generate(message, stream=False, output_format="json")
    result = json.loads(response.read().decode("utf-8"))
    return result["response"]


def ask_router(message):
    """Ask the lightweight local planner to order online actions only."""
    prompt = f"""
You are Jarvis's ONLINE request planner. You have no access to private
memory and must never classify, retrieve, or answer from personal memory.

Return ordered steps using only TOOL or CLOUD. Preserve the user's order.
Use TOOL only for an explicit request to perform a supported action. Merely
asking how to do something is informational and must be CLOUD.

Supported tools:
- GET_TIME: explicitly requested current time; target=null
- OPEN_APP: explicitly requested open/launch/start of chrome, notepad, or
  calculator; target must be that lowercase application name

Everything else is CLOUD, including requests that appear personal. Do not
emit LOCAL, BLOCKED, memory fields, explanations outside JSON, arbitrary
commands, or unsupported tools. For mixed intents, provide one step per
intent in the original order and use only the content for that step.

Return only JSON in this shape:
{{"steps":[{{"type":"TOOL|CLOUD","action":null,"target":null,"content":null}}]}}

User request:
{message}
"""
    response = _ollama_generate(prompt, stream=False, output_format="json")
    result = json.loads(response.read().decode("utf-8"))
    return result["response"]
