import json
import time

from router import route_message


CASES = [
    {
        "message": "Tell me what you remember about my Jarvis and explain how Gemini Live could fit into it.",
        "expected": [
            ("LOCAL", None, None),
            ("CLOUD", None, None),
        ],
    },
    {
        "message": "Explain how Gemini Live could fit into my Jarvis, and then tell me what you remember about my Jarvis.",
        "expected": [
            ("CLOUD", None, None),
            ("LOCAL", None, None),
        ],
    },
    {
        "message": "Open Chrome and explain how Python dictionaries work.",
        "expected": [
            ("TOOL", "OPEN_APP", "chrome"),
            ("CLOUD", None, None),
        ],
    },
    {
        "message": "Explain how Python dictionaries work, then open Chrome.",
        "expected": [
            ("CLOUD", None, None),
            ("TOOL", "OPEN_APP", "chrome"),
        ],
    },
    {
        "message": "Tell me what GPU I have and then open Chrome.",
        "expected": [
            ("LOCAL", None, None),
            ("TOOL", "OPEN_APP", "chrome"),
        ],
    },
    {
        "message": "Open Chrome, then tell me what GPU I have.",
        "expected": [
            ("TOOL", "OPEN_APP", "chrome"),
            ("LOCAL", None, None),
        ],
    },
    {
        "message": "First tell me what you remember about my university, then explain what computer engineering is.",
        "expected": [
            ("LOCAL", None, None),
            ("CLOUD", None, None),
        ],
    },
    {
        "message": "Explain what computer engineering is, then tell me what university I attend.",
        "expected": [
            ("CLOUD", None, None),
            ("LOCAL", None, None),
        ],
    },
]


def normalize(result):
    steps = result.get("steps", [])
    return [
        (
            step.get("type"),
            step.get("action"),
            step.get("target"),
        )
        for step in steps
    ]


print("=" * 60)
print("JARVIS ROUTER ORDERING DIAGNOSTIC")
print("=" * 60)
print(f"Cases: {len(CASES)}")
print()

passed = 0
total_time = 0.0

for index, case in enumerate(CASES, 1):
    message = case["message"]
    expected = case["expected"]

    start = time.perf_counter()

    try:
        result = route_message(message)
        elapsed = time.perf_counter() - start
        actual = normalize(result)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        actual = [("ERROR", None, None)]
        result = {"error": repr(exc)}

    total_time += elapsed

    ok = actual == expected

    if ok:
        passed += 1

    print(f"[{index:02d}/{len(CASES)}] {message}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    print(f"  Expected: {expected}")
    print(f"  Actual:   {actual}")
    print(f"  Time:     {elapsed:.2f}s")

    if not ok:
        print("  RAW:", json.dumps(result, ensure_ascii=False))

    print()

average = total_time / len(CASES)

print("=" * 60)
print(f"RESULT: {passed}/{len(CASES)}")
print(f"ACCURACY: {passed / len(CASES) * 100:.1f}%")
print(f"TOTAL TIME: {total_time:.2f}s")
print(f"AVERAGE: {average:.2f}s/request")
print("=" * 60)
