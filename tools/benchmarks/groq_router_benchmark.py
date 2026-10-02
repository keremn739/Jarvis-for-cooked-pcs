"""Compare Jarvis routing with direct and hybrid Groq classification.

Ground-truth cases in this file are static and manually labelled. Running the
benchmark makes Groq API requests and may also invoke the configured local
router through ``router.route_online``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import router


MODEL = "openai/gpt-oss-20b"
REPORT_PATH = Path("/tmp/jarvis_groq_router_benchmark.json")
CATEGORIES = ("TOOL", "WORK", "CLOUD", "MIXED/AMBIGUOUS")


def _tool(action, target=None):
    return {"type": "TOOL", "action": action, "target": target}


def _work():
    return {"type": "WORK"}


def _cloud():
    return {"type": "CLOUD"}


def _case(case_id, category, message, *expected, language="EN", note=None, switch_to=None):
    return {
        "id": case_id,
        "category": category,
        "language": language,
        "message": message,
        "expected": {"steps": list(expected), "switch_to": switch_to},
        "note": note,
    }


# Static, manually labelled cases. No model is used to produce these labels.
CASES = (
    # TOOL (16)
    _case("T01", "TOOL", "What time is it?", _tool("GET_TIME")),
    _case("T02", "TOOL", "Tell me the current time.", _tool("GET_TIME")),
    _case("T03", "TOOL", "Time right now, please.", _tool("GET_TIME")),
    _case("T04", "TOOL", "Saat kaç?", _tool("GET_TIME"), language="TR"),
    _case("T05", "TOOL", "Open Chrome.", _tool("OPEN_APP", "chrome")),
    _case("T06", "TOOL", "Launch Notepad.", _tool("OPEN_APP", "notepad")),
    _case("T07", "TOOL", "Start the calculator.", _tool("OPEN_APP", "calculator")),
    _case("T08", "TOOL", "Open Google Chrome for me.", _tool("OPEN_APP", "chrome")),
    _case("T09", "TOOL", "Chrome'u aç.", _tool("OPEN_APP", "chrome"), language="TR"),
    _case("T10", "TOOL", "Hesap makinesini aç.", _tool("OPEN_APP", "calculator"), language="TR"),
    _case("T11", "TOOL", "Could you launch Chrome now?", _tool("OPEN_APP", "chrome")),
    _case("T12", "TOOL", "Şu an saat kaç?", _tool("GET_TIME"), language="TR"),
    _case("T13", "TOOL", "Please open the calculator.", _tool("OPEN_APP", "calculator")),
    _case("T14", "TOOL", "Start Notepad so I can take notes.", _tool("OPEN_APP", "notepad")),
    _case("T15", "TOOL", "Can you tell me what time it is?", _tool("GET_TIME")),
    _case("T16", "TOOL", "Open the notepad app.", _tool("OPEN_APP", "notepad")),

    # WORK (16)
    _case("W01", "WORK", "Implement this feature in the project", _work()),
    _case("W02", "WORK", "Implement a login system in my app", _work()),
    _case("W03", "WORK", "Build this application", _work()),
    _case("W04", "WORK", "Build the feature described above", _work()),
    _case("W05", "WORK", "Debug this Python script", _work()),
    _case("W06", "WORK", "Debug the code in this file", _work()),
    _case("W07", "WORK", "Fix this bug in the code", _work()),
    _case("W08", "WORK", "Fix the function in my project", _work()),
    _case("W09", "WORK", "Refactor this function", _work()),
    _case("W10", "WORK", "Refactor the project", _work()),
    _case("W11", "WORK", "Modify this Python file", _work()),
    _case("W12", "WORK", "Delete draft.txt from this software project's source tree", _work(), note="Explicit source-tree context makes this a project task."),
    _case("W13", "WORK", "Projede bu özelliği uygula", _work(), language="TR"),
    _case("W14", "WORK", "Bu Python betiğindeki hatayı düzelt", _work(), language="TR"),
    _case("W15", "WORK", "Test paketini çalıştır", _work(), language="TR"),
    _case("W16", "WORK", "Yeni bir Python betiği oluştur", _work(), language="TR"),

    # CLOUD (16): informational traps, action words, and unsupported requests.
    _case("C01", "CLOUD", "How do I fix this?", _cloud()),
    _case("C02", "CLOUD", "Why is this code broken?", _cloud()),
    _case("C03", "CLOUD", "Can you explain how to fix this?", _cloud()),
    _case("C04", "CLOUD", "What should I fix?", _cloud()),
    _case("C05", "CLOUD", "Create a shopping list", _cloud()),
    _case("C06", "CLOUD", "Fix my Wi-Fi", _cloud()),
    _case("C07", "CLOUD", "Add this to my memory", _cloud()),
    _case("C08", "CLOUD", "Delete the file named draft.txt.", _cloud(), note="No project or software context; this is not a Jarvis-supported file action."),
    _case("C09", "CLOUD", "Build me a workout plan", _cloud()),
    _case("C10", "CLOUD", "How do I open Chrome?", _cloud()),
    _case("C11", "CLOUD", "Open Spotify.", _cloud(), note="OPEN_APP supports only Chrome, Notepad, and Calculator."),
    _case("C12", "CLOUD", "Send an email to my team about the release.", _cloud(), note="Jarvis has no email execution tool."),
    _case("C13", "CLOUD", "Bugün hava nasıl?", _cloud(), language="TR"),
    _case("C14", "CLOUD", "Bu akşam ne pişireyim?", _cloud(), language="TR"),
    _case("C15", "CLOUD", "Wi-Fi bağlantım neden yavaş?", _cloud(), language="TR"),
    _case("C16", "CLOUD", "Chrome neden yavaş?", _cloud(), language="TR"),

    # MIXED/AMBIGUOUS (16): ordered multi-intents and uncertain/unsupported asks.
    _case("M01", "MIXED/AMBIGUOUS", "Open Chrome, then tell me the weather.", _tool("OPEN_APP", "chrome"), _cloud()),
    _case("M02", "MIXED/AMBIGUOUS", "Tell me the weather, then open Chrome.", _cloud(), _tool("OPEN_APP", "chrome")),
    _case("M03", "MIXED/AMBIGUOUS", "Open Chrome, then run the test suite.", _tool("OPEN_APP", "chrome"), _work()),
    _case("M04", "MIXED/AMBIGUOUS", "Run the test suite, then open Chrome.", _work(), _tool("OPEN_APP", "chrome")),
    _case("M05", "MIXED/AMBIGUOUS", "Tell me the weather, then explain Python dictionaries.", _cloud(), _cloud()),
    _case("M06", "MIXED/AMBIGUOUS", "Open Chrome, then open Notepad.", _tool("OPEN_APP", "chrome"), _tool("OPEN_APP", "notepad")),
    _case("M07", "MIXED/AMBIGUOUS", "Open Chrome, then tell me the weather, then run the test suite.", _tool("OPEN_APP", "chrome"), _cloud(), _work()),
    _case("M08", "MIXED/AMBIGUOUS", "Tell me the weather, then open Chrome, and finally run the test suite.", _cloud(), _tool("OPEN_APP", "chrome"), _work()),
    _case("M09", "MIXED/AMBIGUOUS", "Run the test suite, then tell me the weather.", _work(), _cloud()),
    _case("M10", "MIXED/AMBIGUOUS", "Build the feature, then explain Python dictionaries.", _work(), _cloud()),
    _case("M11", "MIXED/AMBIGUOUS", "Tell me the weather, then build the feature.", _cloud(), _work()),
    _case("M12", "MIXED/AMBIGUOUS", "Run the tests, then explain the result format.", _work(), _cloud()),
    _case("M13", "MIXED/AMBIGUOUS", "Chrome'u aç, sonra hava durumunu özetle.", _tool("OPEN_APP", "chrome"), _cloud(), language="TR"),
    _case("M14", "MIXED/AMBIGUOUS", "Önce hava durumunu açıkla, sonra Chrome'u aç.", _cloud(), _tool("OPEN_APP", "chrome"), language="TR"),
    _case("M15", "MIXED/AMBIGUOUS", "Saat kaç, sonra Chrome'u aç?", _tool("GET_TIME"), _tool("OPEN_APP", "chrome"), language="TR"),
    _case("M16", "MIXED/AMBIGUOUS", "Tell me the weather, then run the test suite, and finally open Chrome.", _cloud(), _work(), _tool("OPEN_APP", "chrome")),
)

ENGLISH_CASES = tuple(case for case in CASES if case["language"] == "EN")
TURKISH_CASES = tuple(case for case in CASES if case["language"] == "TR")


STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["TOOL", "WORK", "CLOUD"]},
        "action": {
            "anyOf": [
                {"type": "string", "enum": ["GET_TIME", "OPEN_APP"]},
                {"type": "null"},
            ]
        },
        "target": {
            "anyOf": [
                {"type": "string", "enum": ["chrome", "notepad", "calculator"]},
                {"type": "null"},
            ]
        },
        "content": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["type", "action", "target", "content"],
    "additionalProperties": False,
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": STEP_SCHEMA},
        "switch_to": {
            "anyOf": [
                {"type": "string", "enum": ["LOCAL", "ONLINE"]},
                {"type": "null"},
            ]
        },
    },
    "required": ["steps", "switch_to"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are Jarvis's online router. Classify each intent using this contract:
- Answering a question about how to do something is CLOUD.
- Asking Jarvis to perform a supported action is TOOL.
- Asking Jarvis to modify, build, debug, or refactor software is WORK.
- Use CLOUD for informational content requests and unsupported actions.
- For multiple intents, emit one step per intent in exact original order.
- Never collapse multiple intents into one step or stop after the first intent.
- Never invent unsupported tools. OPEN_APP supports only chrome, notepad,
  and calculator. GET_TIME is the only other supported tool.
- A plain file operation without explicit software/project context is CLOUD;
  project source-code changes or operations are WORK.

Output ONLY the strict JSON schema. Use only TOOL, WORK, and CLOUD step types.
For TOOL, set action and target correctly; GET_TIME target is null. Set action
and target to null for WORK and CLOUD. Put the relevant request text in content
or null if not needed. Use switch_to only for an exact Jarvis mode command; in
that case return no steps and the requested mode. Otherwise switch_to is null."""


def _signature(plan):
    """Reduce a router plan to routing decisions, ignoring copied content."""
    if not isinstance(plan, dict):
        return None
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return None
    signature = []
    for step in steps:
        if not isinstance(step, dict):
            return None
        step_type = str(step.get("type", "")).strip().upper()
        if step_type == "TOOL":
            signature.append(
                (
                    "TOOL",
                    str(step.get("action") or "").upper(),
                    str(step["target"]).lower() if step.get("target") is not None else None,
                )
            )
        elif step_type in {"WORK", "CLOUD"}:
            signature.append((step_type,))
        else:
            return None
    switch_to = plan.get("switch_to")
    return {"steps": signature, "switch_to": switch_to}


def _expected_signature(case):
    return _signature(case["expected"])


def _read_response_content(response):
    content = response.choices[0].message.content
    if isinstance(content, str):
        return json.loads(content)
    if isinstance(content, dict):
        return content
    raise ValueError("Groq returned no JSON response content")


def _token_count(usage, *names):
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return 0


def _latency_summary(latencies):
    if not latencies:
        return {"mean_seconds": None, "p50_seconds": None, "p95_seconds": None}
    ordered = sorted(latencies)

    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "mean_seconds": statistics.mean(ordered),
        "p50_seconds": percentile(0.50),
        "p95_seconds": percentile(0.95),
    }


class GroqPacer:
    def __init__(self, delay_seconds):
        self.delay_seconds = delay_seconds
        self.last_call_finished = None

    def wait(self):
        if self.last_call_finished is not None:
            remaining = self.delay_seconds - (time.monotonic() - self.last_call_finished)
            if remaining > 0:
                time.sleep(remaining)

    def mark_finished(self):
        self.last_call_finished = time.monotonic()


class GroqRunner:
    def __init__(self, client, pacer):
        self.client = client
        self.pacer = pacer
        self.metrics = {
            "calls": 0,
            "api_failures": 0,
            "latencies_seconds": [],
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
        }

    def classify(self, message):
        self.pacer.wait()

        self.metrics["calls"] += 1
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "jarvis_router_plan",
                        "strict": True,
                        "schema": PLAN_SCHEMA,
                    },
                },
            )
            plan = _read_response_content(response)
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.metrics["tokens"]["prompt"] += _token_count(
                    usage, "prompt_tokens", "input_tokens"
                )
                self.metrics["tokens"]["completion"] += _token_count(
                    usage, "completion_tokens", "output_tokens"
                )
                self.metrics["tokens"]["total"] += _token_count(usage, "total_tokens")
            return plan, None
        except Exception as error:  # Record API, schema, and response parsing failures.
            self.metrics["api_failures"] += 1
            return None, f"{type(error).__name__}: {error}"
        finally:
            elapsed = time.perf_counter() - started
            self.metrics["latencies_seconds"].append(elapsed)
            self.pacer.mark_finished()


def _hybrid_deterministic(message):
    mode = router.requested_mode(message)
    if mode is not None:
        return {"steps": [], "switch_to": mode}

    work_step = router.match_work(message)
    if work_step:
        return {"steps": [work_step], "switch_to": None}

    split_steps = router._split_ordered_intents(message)
    if split_steps:
        return {"steps": split_steps, "switch_to": None}

    tool_step = router.match_tool(message)
    if tool_step:
        return {"steps": [tool_step], "switch_to": None}

    return None


def _evaluate_current(cases):
    predictions = []
    for case in cases:
        error = None
        try:
            plan = router.route_online(case["message"])
        except Exception as exception:
            plan = None
            error = f"{type(exception).__name__}: {exception}"
        predictions.append(_prediction_record(case, plan, error, path="CURRENT router"))
    return predictions


def _prediction_record(case, plan, error=None, path=None):
    actual = _signature(plan) if plan is not None else None
    expected = _expected_signature(case)
    return {
        "case_id": case["id"],
        "category": case["category"],
        "language": case["language"],
        "message": case["message"],
        "expected": expected,
        "actual": actual,
        "correct": actual == expected and error is None,
        "error": error,
        "path": path,
        "note": case.get("note"),
    }


def _evaluate_groq_direct(cases, runner):
    predictions = []
    for case in cases:
        plan, error = runner.classify(case["message"])
        predictions.append(_prediction_record(case, plan, error, path=MODEL))
    return predictions


def _evaluate_hybrid(cases, runner):
    predictions = []
    deterministic_cases = 0
    for case in cases:
        plan = _hybrid_deterministic(case["message"])
        error = None
        if plan is None:
            plan, error = runner.classify(case["message"])
            path = f"Groq fallback ({MODEL})"
        else:
            deterministic_cases += 1
            path = "deterministic shortcut"
        predictions.append(_prediction_record(case, plan, error, path=path))
    return predictions, deterministic_cases


def _accuracy_report(cases, predictions):
    by_id = {prediction["case_id"]: prediction for prediction in predictions}
    correct = sum(prediction["correct"] for prediction in predictions)
    category_stats = {}
    for category in CATEGORIES:
        members = [case for case in cases if case["category"] == category]
        category_correct = sum(by_id[case["id"]]["correct"] for case in members)
        category_stats[category] = {
            "correct": category_correct,
            "total": len(members),
            "accuracy": category_correct / len(members) if members else None,
        }
    language_stats = {}
    for language, case_language in (("ENGLISH", "EN"), ("TURKISH", "TR")):
        members = [case for case in cases if case["language"] == case_language]
        language_correct = sum(by_id[case["id"]]["correct"] for case in members)
        language_stats[language] = {
            "correct": language_correct,
            "total": len(members),
            "accuracy": language_correct / len(members) if members else None,
        }
    return {
        "correct": correct,
        "total": len(cases),
        "accuracy": correct / len(cases) if cases else None,
        "english": language_stats["ENGLISH"],
        "turkish": language_stats["TURKISH"],
        "english_gate_passed": (
            language_stats["ENGLISH"]["accuracy"] == 1.0
            if language_stats["ENGLISH"]["total"]
            else False
        ),
        "mixed_multi_intent": category_stats["MIXED/AMBIGUOUS"],
        "by_category": category_stats,
        "by_language": language_stats,
        "failed_cases": [prediction for prediction in predictions if not prediction["correct"]],
    }


def _hybrid_path_report(predictions):
    report = {}
    for path, label in (
        ("deterministic shortcut", "deterministic_shortcut"),
        (f"Groq fallback ({MODEL})", "groq_fallback"),
    ):
        selected = [prediction for prediction in predictions if prediction["path"] == path]
        correct = sum(prediction["correct"] for prediction in selected)
        report[label] = {
            "correct": correct,
            "incorrect": len(selected) - correct,
            "total": len(selected),
            "accuracy": correct / len(selected) if selected else None,
        }
    return report


def _groq_metrics(runner):
    metrics = runner.metrics
    return {
        "calls": metrics["calls"],
        "api_failures": metrics["api_failures"],
        "latency": _latency_summary(metrics["latencies_seconds"]),
        "tokens": metrics["tokens"],
    }


def _format_accuracy(metric):
    accuracy = metric["accuracy"]
    if accuracy is None:
        return f"{metric['correct']}/{metric['total']} (n/a)"
    return f"{metric['correct']}/{metric['total']} ({accuracy:.1%})"


def _check_case_counts():
    counts = {category: sum(case["category"] == category for case in CASES) for category in CATEGORIES}
    expected_counts = {category: 16 for category in CATEGORIES}
    if counts != expected_counts or len(CASES) != 64:
        raise RuntimeError(f"Benchmark corpus count error: {counts}")
    if len(ENGLISH_CASES) + len(TURKISH_CASES) != len(CASES):
        raise RuntimeError("Every benchmark case must have an EN or TR language label")
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=len(CASES), help="number of leading cases to run")
    parser.add_argument("--start", type=int, default=0, help="starting case index")
    parser.add_argument("--delay", type=float, default=2.1, help="minimum delay between Groq calls")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.delay < 0:
        parser.error("--delay must be non-negative")
    _check_case_counts()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        parser.error("GROQ_API_KEY must be set in the environment")

    try:
        from groq import Groq
    except ImportError as error:
        parser.error(f"Groq Python package is required to run the benchmark: {error}")

    selected_cases = list(CASES[args.start : args.start + args.limit])
    client = Groq(api_key=api_key)
    pacer = GroqPacer(args.delay)
    direct_runner = GroqRunner(client, pacer)
    hybrid_runner = GroqRunner(client, pacer)

    current_predictions = _evaluate_current(selected_cases)
    direct_predictions = _evaluate_groq_direct(selected_cases, direct_runner)
    hybrid_predictions, deterministic_cases = _evaluate_hybrid(selected_cases, hybrid_runner)

    results = {
        "CURRENT": _accuracy_report(selected_cases, current_predictions),
        "GROQ_DIRECT": _accuracy_report(selected_cases, direct_predictions),
        "HYBRID": _accuracy_report(selected_cases, hybrid_predictions),
    }
    hybrid_path_diagnostics = _hybrid_path_report(hybrid_predictions)
    report = {
        "model": MODEL,
        "case_counts": _check_case_counts(),
        "cases_run": len(selected_cases),
        "delay_seconds": args.delay,
        "results": results,
        "groq_metrics": {
            "GROQ_DIRECT": _groq_metrics(direct_runner),
            "HYBRID": _groq_metrics(hybrid_runner),
        },
        "hybrid_groq_calls": hybrid_runner.metrics["calls"],
        "hybrid_total_cases": len(selected_cases),
        "hybrid_deterministic_cases": deterministic_cases,
        "hybrid_groq_call_ratio": (
            hybrid_runner.metrics["calls"] / len(selected_cases)
            if selected_cases
            else None
        ),
        "hybrid_path_diagnostics": hybrid_path_diagnostics,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Groq router benchmark ({MODEL}) — {len(selected_cases)} cases")
    print(f"Report: {REPORT_PATH}")
    for name, result in results.items():
        print(
            f"{name}: TOTAL {_format_accuracy(result)}; "
            f"ENGLISH {_format_accuracy(result['english'])}; "
            f"MIXED/MULTI-INTENT {_format_accuracy(result['mixed_multi_intent'])}; "
            f"ENGLISH GATE={'PASS' if result['english_gate_passed'] else 'FAIL'}"
        )
        counts = [
            ("TOTAL", result),
            ("ENGLISH", result["by_language"]["ENGLISH"]),
            ("TURKISH", result["by_language"]["TURKISH"]),
            *((category, result["by_category"][category]) for category in CATEGORIES),
        ]
        print(
            f"{name} breakdown: "
            + " | ".join(
                f"{label}: {metric['correct']}/{metric['total']}"
                for label, metric in counts
            )
        )
    for name, metrics in report["groq_metrics"].items():
        latency = metrics["latency"]
        mean = latency["mean_seconds"]
        p50 = latency["p50_seconds"]
        p95 = latency["p95_seconds"]
        if mean is None:
            print(f"{name} Groq: 0 calls; tokens={metrics['tokens']}; API failures={metrics['api_failures']}")
        else:
            print(
                f"{name} Groq: {metrics['calls']} calls; "
                f"latency mean/p50/p95={mean:.2f}/{p50:.2f}/{p95:.2f}s; "
                f"tokens={metrics['tokens']}; API failures={metrics['api_failures']}"
            )
    print(
        f"HYBRID Groq calls: {hybrid_runner.metrics['calls']}/"
        f"{len(selected_cases)} cases"
    )
    for path, metrics in hybrid_path_diagnostics.items():
        print(
            f"HYBRID {path}: correct {metrics['correct']}, "
            f"incorrect {metrics['incorrect']}, total {metrics['total']}"
        )

    for name, result in results.items():
        for failed in result["failed_cases"]:
            print(
                f"FAIL {name} {failed['case_id']} [{failed['category']}]: "
                f"input={failed['message']!r}; expected={failed['expected']}; "
                f"actual={failed['actual']}; error={failed['error']}"
                f"; model/path={failed['path']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
