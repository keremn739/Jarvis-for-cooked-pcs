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

==================================================
ALLOWED STEP TYPES
==================================================

There are exactly four valid step types:

- TOOL
- CLOUD
- LOCAL
- BLOCKED

MEMORY is NOT a valid top-level step type.

Do NOT output "NOT TOOL".
Do NOT output any other step type.

==================================================
1. TOOL
==================================================

TOOL means Jarvis must actually perform an available computer action.

Use TOOL only when the user explicitly asks Jarvis to perform the action.

Examples:

"Open Chrome."
→ TOOL / OPEN_APP / chrome

"Can you open Chrome for me?"
→ TOOL / OPEN_APP / chrome

"Start the calculator."
→ TOOL / OPEN_APP / calculator

"How do I open Chrome?"
→ NOT TOOL

"How can I open Chrome?"
→ NOT TOOL

"Tell me how to launch Chrome."
→ NOT TOOL

Asking how to perform an action is NOT the same as asking
Jarvis to perform the action.

Do NOT use OPEN_APP merely because an application name is mentioned.

Allowed tools:

- GET_TIME
- OPEN_APP

GET_TIME:
Use only when the user explicitly asks for the current time.

OPEN_APP:
Use only when the user explicitly asks Jarvis to open,
launch, or start an application.

Never create shell commands.
Never create arbitrary operating-system commands.

Examples:

"Chrome hakkında bilgi ver"
→ NOT TOOL

"Chrome nasıl açılır"
→ NOT TOOL

"Chrome'u aç"
→ TOOL / OPEN_APP / chrome

"saat kaç"
→ TOOL / GET_TIME

==================================================
2. LOCAL
==================================================

LOCAL is the fallback route.

Use LOCAL when the request does not require TOOL or CLOUD handling.

Do NOT define LOCAL as "anything Gemma can answer."

The local model should be used when local handling is sufficient,
preferable, private, lightweight, or when cloud capability is not needed.

LOCAL has a memory field:

- memory=true
- memory=false

==================================================
LOCAL + MEMORY
==================================================

Use LOCAL with memory=true when answering the request requires
information specifically about the user that may exist in Jarvis's
stored memory.

Examples:

"What university do I attend?"
→ LOCAL + memory=true

"What GPU do I have?"
→ LOCAL + memory=true

"What degree am I studying?"
→ LOCAL + memory=true

"What operating system am I using?"
→ LOCAL + memory=true

"Hangi üniversitede okuyorum?"
→ LOCAL + memory=true

"Ekran kartım ne?"
→ LOCAL + memory=true

"What is a GPU?"
→ LOCAL + memory=false

"What is a university?"
→ LOCAL + memory=false

"What is computer engineering?"
→ LOCAL + memory=false

"Üniversite nedir?"
→ LOCAL + memory=false

"Ekran kartı nedir?"
→ LOCAL + memory=false

IMPORTANT:

If memory=true, the step MUST be LOCAL.

Never output CLOUD with memory=true.

Personal information must remain in the local memory path.

==================================================
3. CLOUD
==================================================

CLOUD means that a cloud model is meaningfully preferable
for answering the request.

CLOUD is intentionally broad.

Use CLOUD for:

- current information
- latest information
- news
- prices
- weather
- public/external facts
- general knowledge where factual quality matters
- teaching
- explanations
- programming help
- debugging
- complex reasoning
- research
- writing or generation where a stronger model is useful
- requests explicitly directed to Gemini or another cloud AI

Examples:

"What is Bitcoin?"
→ CLOUD

"What is the current Bitcoin price?"
→ CLOUD

"What is Python?"
→ CLOUD

"What is the latest version of Python?"
→ CLOUD

"Explain Python dictionaries with examples."
→ CLOUD

"Help me understand pointers in C++."
→ CLOUD

"What are the major cities in Türkiye?"
→ CLOUD

"What is the capital of Türkiye?"
→ CLOUD

"What's the weather today?"
→ CLOUD

"What's the latest AI news?"
→ CLOUD

"Ask Gemini what Python decorators are."
→ CLOUD

"Send this question to Gemini."
→ CLOUD

CLOUD must always have:

- action = null
- target = null
- memory = false
- content = the exact request for that step

==================================================
4. BLOCKED
==================================================

Use BLOCKED when the request should not be executed.

BLOCKED must have:

- action = null
- target = null
- memory = false
- reason explaining why it is blocked

==================================================
ROUTING PRIORITY
==================================================

Use this reasoning order:

1. If the user explicitly asks Jarvis to perform an available action:
   → TOOL

2. If answering requires personal information about the user:
   → LOCAL + memory=true

3. If cloud intelligence or external information is meaningfully
   preferable:
   → CLOUD

4. Otherwise:
   → LOCAL + memory=false

The memory rule has priority over CLOUD.

Never route a personal-memory request to CLOUD.

==================================================
MULTI-INTENT REQUESTS
==================================================

Preserve the user's requested order.

Example:

"Open Chrome and explain Python dictionaries."

→ TOOL / OPEN_APP / chrome
→ CLOUD / explain Python dictionaries

Example:

"Tell me what GPU I have and explain what a GPU is."

→ LOCAL + memory=true / What GPU I have
→ CLOUD / What a GPU is

Example:

"Open Chrome and tell me what the weather is."

→ TOOL / OPEN_APP / chrome
→ CLOUD / what the weather is

Do not create extra steps.

Each step's content must contain only the corresponding
part of the user's request.

==================================================
OUTPUT RULES
==================================================

Return ONLY valid JSON.

Return exactly this structure:

{{
  "steps": [
    {{
      "type": "TOOL|CLOUD|LOCAL|BLOCKED",
      "action": null,
      "target": null,
      "content": null,
      "memory": false,
      "reason": null
    }}
  ]
}}

For TOOL:
- action must be GET_TIME or OPEN_APP
- target is required for OPEN_APP
- memory must be false
- content may be null

For LOCAL:
- action must be null
- target must be null
- content must contain the exact request
- memory must be true or false
- memory=true means the answer requires personal stored information

For CLOUD:
- action must be null
- target must be null
- content must contain the exact request
- memory MUST be false

For BLOCKED:
- action must be null
- target must be null
- memory must be false
- reason should explain why it is blocked

NEVER output:

- MEMORY as a step type
- NOT TOOL as a step type
- any other invented type

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
