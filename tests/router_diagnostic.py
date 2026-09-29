import json
import time

from router import route_message


CASES = [
    # ─────────────────────────────────────────────
    # TOOL vs INSTRUCTION
    # ─────────────────────────────────────────────

    {
        "id": 1,
        "message": "Open Chrome.",
        "expected": [
            ("TOOL", "OPEN_APP", "chrome"),
        ],
    },
    {
        "id": 2,
        "message": "How do I open Chrome?",
        "expected": [
            ("LOCAL", None, None),
        ],
    },
    {
        "id": 3,
        "message": "Can you open Chrome for me?",
        "expected": [
            ("TOOL", "OPEN_APP", "chrome"),
        ],
    },
    {
        "id": 4,
        "message": "Tell me how to launch Chrome.",
        "expected": [
            ("LOCAL", None, None),
        ],
    },

    # ─────────────────────────────────────────────
    # LOCAL MEMORY vs GENERAL LOCAL/CLOUD
    # ─────────────────────────────────────────────

    {
        "id": 5,
        "message": "What GPU do I have?",
        "expected": [
            ("LOCAL", None, None),
        ],
    },
    {
        "id": 6,
        "message": "What is a GPU?",
        "expected": [
            ("CLOUD", None, None),
        ],
    },
    {
        "id": 7,
        "message": "What university do I attend?",
        "expected": [
            ("LOCAL", None, None),
        ],
    },
    {
        "id": 8,
        "message": "What is computer engineering?",
        "expected": [
            ("CLOUD", None, None),
        ],
    },

    # ─────────────────────────────────────────────
    # CLOUD BOUNDARIES
    # ─────────────────────────────────────────────

    {
        "id": 9,
        "message": "What's the latest AI news?",
        "expected": [
            ("CLOUD", None, None),
        ],
    },
    {
        "id": 10,
        "message": "Explain Python dictionaries with examples.",
        "expected": [
            ("CLOUD", None, None),
        ],
    },
    {
        "id": 11,
        "message": "What do you remember about my Jarvis project?",
        "expected": [
            ("LOCAL", None, None),
        ],
    },
    {
        "id": 12,
        "message": "How can I improve my Jarvis router?",
        "expected": [
            ("CLOUD", None, None),
        ],
    },

    # ─────────────────────────────────────────────
    # MULTI-INTENT
    # ─────────────────────────────────────────────

    {
        "id": 13,
        "message": "Open Chrome and explain Python dictionaries.",
        "expected": [
            ("TOOL", "OPEN_APP", "chrome"),
            ("CLOUD", None, None),
        ],
    },
    {
        "id": 14,
        "message": "Tell me what GPU I have and explain what a GPU is.",
        "expected": [
            ("LOCAL", None, None),
            ("CLOUD", None, None),
        ],
    },
    {
        "id": 15,
        "message": "Open Chrome and tell me what the weather is.",
        "expected": [
            ("TOOL", "OPEN_APP", "chrome"),
            ("CLOUD", None, None),
        ],
    },
    {
        "id": 16,
        "message": "Tell me what you remember about my Jarvis and explain how Gemini Live could fit into it.",
        "expected": [
            ("LOCAL", None, None),
            ("CLOUD", None, None),
        ],
    },
]


def normalize(result):
    """
    Convert the router result into the fields we actually care about
    for this diagnostic benchmark.
    """
    steps = result.get("steps", [])

    normalized = []

    for step in steps:
        normalized.append(
            (
                step.get("type"),
                step.get("action"),
                step.get("target"),
            )
        )

    return normalized


def main():
    print("=" * 60)
    print("JARVIS ROUTER DIAGNOSTIC")
    print("=" * 60)
    print(f"Cases: {len(CASES)}")
    print()

    passed = 0
    total_time = 0.0

    for case in CASES:
        print(f"[{case['id']:02d}/{len(CASES)}] {case['message']}")

        start = time.perf_counter()

        try:
            result = route_message(case["message"])
            elapsed = time.perf_counter() - start
            total_time += elapsed

            actual = normalize(result)
            expected = case["expected"]

            ok = actual == expected

            if ok:
                passed += 1
                status = "PASS"
            else:
                status = "FAIL"

            print(f"  {status}")
            print(f"  Expected: {expected}")
            print(f"  Actual:   {actual}")
            print(f"  Time:     {elapsed:.2f}s")

            if not ok:
                print(f"  RAW:      {json.dumps(result, ensure_ascii=False)}")

        except Exception as exc:
            elapsed = time.perf_counter() - start
            total_time += elapsed

            print("  ERROR")
            print(f"  {type(exc).__name__}: {exc}")
            print(f"  Time: {elapsed:.2f}s")

        print()

    print("=" * 60)
    print(f"RESULT: {passed}/{len(CASES)}")
    print(f"ACCURACY: {passed / len(CASES) * 100:.1f}%")
    print(f"TOTAL TIME: {total_time:.2f}s")
    print(f"AVERAGE: {total_time / len(CASES):.2f}s/request")
    print("=" * 60)


if __name__ == "__main__":
    main()
