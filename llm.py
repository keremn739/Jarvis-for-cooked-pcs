import urllib.request
import json
import time


def ask_llm(message):

    data = {
        "model": "gemma3:4b",
        "prompt": message,
        "stream": True,
        "think": False
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    start = time.time()

    response = urllib.request.urlopen(request)

    answer = ""
    first_token_time = None

    while True:
        line = response.readline()

        if not line:
            break

        result = json.loads(line)

        if first_token_time is None and result["response"]:
            first_token_time = time.time()

        answer += result["response"]
        print(result["response"], end="", flush=True)

        if result["done"]:
            break

    end = time.time()

    print()
    print(f"\n[Time to first token: {first_token_time - start:.2f} seconds]")
    print(f"[Total response time: {end - start:.2f} seconds]")

    return answer


def ask_llm_json(message):

    data = {
        "model": "gemma3:4b",
        "prompt": message,
        "stream": False,
        "format": "json",
        "think": False
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    response = urllib.request.urlopen(request)

    result = json.loads(response.read().decode("utf-8"))

    return result["response"]


def ask_router(message):

    prompt = f"""
You are the routing and planning brain of a personal AI assistant.

Analyze the user's request semantically and create an ordered execution plan.

The user may speak naturally in Turkish or English.
The input may be lowercase, contain spelling mistakes, omit punctuation,
or look like speech-to-text output.

You MUST preserve the order of the user's requested actions.

Allowed step types:

- MEMORY
  Use when the user is asking about information stored in the user's memory.
  The content field MUST contain the exact part of the user's request
  that should be answered using memory.

- LOCAL
  Use when the local LLM should answer an informational, conversational,
  educational, or programming question.

- CLOUD
  Use when the request requires cloud capabilities or external/current
  information.

- TOOL
  Use when the assistant must actually perform one of the allowed tools.

- BLOCKED
  Use when a request should not be executed.

Allowed tools:

- GET_TIME
- OPEN_APP

TOOL rules:

- GET_TIME means the user explicitly wants to know the current time.
- OPEN_APP means the user explicitly wants the assistant to open an application.
- OPEN_APP target must contain the application name.
- Never create shell commands.
- Never create arbitrary operating-system commands.

IMPORTANT SEMANTIC DISTINCTIONS:

"chrome aç"
→ TOOL / OPEN_APP / chrome

"chrome nasıl açılır"
→ LOCAL

"chrome aç sonra python dictionary nedir"
→
1. TOOL / OPEN_APP / chrome
2. LOCAL / python dictionary nedir

"saat kaç"
→ TOOL / GET_TIME

"ben hangi üniversitede okuyorum"
→ MEMORY

"python dictionary nedir"
→ LOCAL

"bugün hava nasıl"
→ CLOUD

"python dictionary nedir ve nasıl kullanılır"
→ LOCAL

Do NOT create extra steps.

Do NOT invent actions that the user did not request.

Do NOT use GET_TIME unless the user explicitly asks about the time.

Do NOT use OPEN_APP merely because an application name is mentioned.

For example:

"Chrome'da Python öğrenmek için ne yapmalıyım?"
→ LOCAL

"Chrome'u aç"
→ TOOL / OPEN_APP / chrome

"Chrome hakkında bilgi ver"
→ LOCAL

The action and type fields MUST agree.

Valid examples:

TOOL + GET_TIME
TOOL + OPEN_APP

Invalid examples:

LOCAL + OPEN_APP
LOCAL + GET_TIME
CLOUD + OPEN_APP
CLOUD + GET_TIME
MEMORY + OPEN_APP
MEMORY + GET_TIME

IMPORTANT TOOL SEMANTICS:

OPEN_APP is ONLY for explicitly opening or launching an application.

Mentioning an application name does NOT mean OPEN_APP.

"Geminiye gönder"
→ This is NOT OPEN_APP.

"Gemini'ye sor"
→ This is NOT OPEN_APP.

"Gemini'ye gönder"
→ CLOUD if the request is otherwise safe.

"Gemini'yi aç"
→ TOOL / OPEN_APP / Gemini

"Chrome'u aç"
→ TOOL / OPEN_APP / chrome

"Chrome hakkında bilgi ver"
→ LOCAL

"Gemini'ye bu metni gönder"
→ CLOUD

Never use OPEN_APP merely because an application name appears in the request.

For LOCAL, MEMORY and CLOUD steps:

- action must be null
- target must be null unless explicitly needed by the schema
- content MUST contain the exact user request that this step should answer
- Do not leave content null for LOCAL, MEMORY or CLOUD steps.
- For multi-intent requests, each step's content must contain only the
  corresponding part of the user's request.

For TOOL steps:
- action must be one of the allowed tools
- target should contain the tool target when needed
- content can be null

For BLOCKED steps:
- action must be null
- target must be null
- reason should explain why the step is blocked

Return ONLY valid JSON.

Return exactly this structure:

{{
  "steps": [
    {{
      "type": "MEMORY|LOCAL|CLOUD|TOOL|BLOCKED",
      "action": null,
      "target": null,
      "content": null,
      "reason": null
    }}
  ]
}}

User message:
{message}
"""

    data = {
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think": False
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    response = urllib.request.urlopen(request)

    result = json.loads(response.read().decode("utf-8"))

    return result["response"]

