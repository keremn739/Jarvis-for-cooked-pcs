"""Baseline benchmark for JARVIS's current semantic router.

This runner calls ``route_message`` and deliberately never calls the execution
layer. It compares normalized structured plans, rather than raw JSON strings.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from router import route_message


def tool(action, target=None):
    return [{"type": "TOOL", "action": action, "target": target}]


def route(route_type):
    return [{"type": route_type}]


# Case 053 from the supplied specification is incomplete, so it is not included.
CASES = [
    ("001", "GET_TIME", "What time is it?", tool("GET_TIME")),
    ("002", "GET_TIME", "Tell me the current time.", tool("GET_TIME")),
    ("003", "GET_TIME", "What's the time right now?", tool("GET_TIME")),
    ("004", "GET_TIME", "Can you tell me the time?", tool("GET_TIME")),
    ("005", "GET_TIME", "saat kaç?", tool("GET_TIME")),
    ("006", "GET_TIME", "şu an saat kaç", tool("GET_TIME")),
    ("007", "GET_TIME", "bana saati söyler misin", tool("GET_TIME")),
    ("008", "GET_TIME", "what's the current time", tool("GET_TIME")),
    ("009", "GET_TIME", "time?", tool("GET_TIME")),
    ("010", "GET_TIME", "could you check the time for me", tool("GET_TIME")),
    ("011", "OPEN_APP", "Open Chrome.", tool("OPEN_APP", "chrome")),
    ("012", "OPEN_APP", "Open Google Chrome.", tool("OPEN_APP", "chrome")),
    ("013", "OPEN_APP", "Launch Chrome.", tool("OPEN_APP", "chrome")),
    ("014", "OPEN_APP", "Start Chrome.", tool("OPEN_APP", "chrome")),
    ("015", "OPEN_APP", "chrome aç", tool("OPEN_APP", "chrome")),
    ("016", "OPEN_APP", "Google Chrome'u aç.", tool("OPEN_APP", "chrome")),
    ("017", "OPEN_APP", "Open Notepad.", tool("OPEN_APP", "notepad")),
    ("018", "OPEN_APP", "Launch Notepad.", tool("OPEN_APP", "notepad")),
    ("019", "OPEN_APP", "notepad aç", tool("OPEN_APP", "notepad")),
    ("020", "OPEN_APP", "Open the calculator.", tool("OPEN_APP", "calculator")),
    ("021", "OPEN_APP", "Launch calculator.", tool("OPEN_APP", "calculator")),
    ("022", "OPEN_APP", "hesap makinesini aç", tool("OPEN_APP", "calculator")),
    ("023", "OPEN_APP", "start the calculator", tool("OPEN_APP", "calculator")),
    ("024", "OPEN_APP", "Can you open Chrome for me?", tool("OPEN_APP", "chrome")),
    ("025", "OPEN_APP", "Please launch the calculator.", tool("OPEN_APP", "calculator")),
    ("026", "LOCAL", "What is Python?", route("LOCAL")),
    ("027", "LOCAL", "Explain what a Python function is.", route("LOCAL")),
    ("028", "LOCAL", "How does a for loop work?", route("LOCAL")),
    ("029", "LOCAL", "What is recursion?", route("LOCAL")),
    ("030", "LOCAL", "Explain object oriented programming.", route("LOCAL")),
    ("031", "LOCAL", "What is an API?", route("LOCAL")),
    ("032", "LOCAL", "What is SQLite?", route("LOCAL")),
    ("033", "LOCAL", "Explain what a database is.", route("LOCAL")),
    ("034", "LOCAL", "How does RAM work?", route("LOCAL")),
    ("035", "LOCAL", "What is a CPU?", route("LOCAL")),
    ("036", "LOCAL", "What is the difference between RAM and storage?", route("LOCAL")),
    ("037", "LOCAL", "Explain TCP in simple terms.", route("LOCAL")),
    ("038", "LOCAL", "What is HTTP?", route("LOCAL")),
    ("039", "LOCAL", "How does DNS work?", route("LOCAL")),
    ("040", "LOCAL", "What does a GPU do?", route("LOCAL")),
    ("041", "LOCAL", "Python'da liste nedir?", route("LOCAL")),
    ("042", "LOCAL", "Python'da dictionary nasıl çalışır?", route("LOCAL")),
    ("043", "LOCAL", "recursion nedir?", route("LOCAL")),
    ("044", "LOCAL", "SQLite ne işe yarar?", route("LOCAL")),
    ("045", "LOCAL", "RAM ile SSD arasındaki fark nedir?", route("LOCAL")),
    ("046", "CLOUD", "What's the weather today?", route("CLOUD")),
    ("047", "CLOUD", "What's the weather right now?", route("CLOUD")),
    ("048", "CLOUD", "What's happening in the markets today?", route("CLOUD")),
    ("049", "CLOUD", "What's the latest news about NVIDIA?", route("CLOUD")),
    ("050", "CLOUD", "What is the current price of Bitcoin?", route("CLOUD")),
    ("051", "CLOUD", "What's the latest version of Python?", route("CLOUD")),
    ("052", "CLOUD", "What happened in tech today?", route("CLOUD")),
]


def normalize_plan(plan):
    """Keep only fields that are part of each test's routing contract."""
    normalized_steps = []
    for step in plan.get("steps", []):
        route_type = str(step.get("type", "")).upper()
        normalized = {"type": route_type}
        if route_type == "TOOL":
            normalized["action"] = str(step.get("action", "")).upper()
            target = step.get("target")
            normalized["target"] = None if target is None else str(target).lower()
        normalized_steps.append(normalized)
    return normalized_steps


def run_benchmark():
    results = []
    category_totals = defaultdict(lambda: {"passed": 0, "total": 0})

    for case_id, category, message, expected in CASES:
        category_totals[category]["total"] += 1
        try:
            actual_plan = route_message(message)
            actual = normalize_plan(actual_plan)
            error = None
        except Exception as exception:  # Record an unavailable router as a failure.
            actual_plan = None
            actual = None
            error = f"{type(exception).__name__}: {exception}"

        passed = actual == expected
        if passed:
            category_totals[category]["passed"] += 1
        results.append(
            {
                "id": case_id,
                "category": category,
                "input": message,
                "expected": expected,
                "actual": actual,
                "actual_plan": actual_plan,
                "passed": passed,
                "error": error,
            }
        )

    total_passed = sum(item["passed"] for item in results)
    routed = sum(item["error"] is None for item in results)
    unavailable = len(CASES) - routed
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_count": len(CASES),
        "routed": routed,
        "unavailable": unavailable,
        "passed": total_passed,
        "failed": routed - total_passed,
        "accuracy": total_passed / routed if routed else None,
        "by_category": dict(category_totals),
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "tests" / "results" / "routing_baseline.json",
    )
    arguments = parser.parse_args()

    report = run_benchmark()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"Cases: {report['case_count']}")
    print(f"Routed: {report['routed']}")
    print(f"Unavailable: {report['unavailable']}")
    print(f"Passed: {report['passed']}")
    print(f"Failed: {report['failed']}")
    accuracy = "N/A" if report["accuracy"] is None else f"{report['accuracy']:.1%}"
    print(f"Accuracy: {accuracy}")
    for category, totals in report["by_category"].items():
        print(f"{category}: {totals['passed']}/{totals['total']}")
    print(f"Report: {arguments.output}")


if __name__ == "__main__":
    main()
