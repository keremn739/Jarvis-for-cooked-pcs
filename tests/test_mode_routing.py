import sys
import types
import unittest
from unittest.mock import patch


if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.float32 = object()
    numpy_stub.array = lambda value, dtype=None: value
    numpy_stub.frombuffer = lambda value, dtype=None: value
    numpy_stub.dot = lambda left, right: 0.0
    sys.modules["numpy"] = numpy_stub

import router
from mode import Mode, ModeState


class ModeRoutingTests(unittest.TestCase):
    def test_work_matcher_requires_explicit_software_context(self):
        positive = (
            "Implement this feature in the project",
            "Debug this Python script",
            "Fix this bug in the code",
            "Refactor this function",
            "Run the test suite",
            "Write a Python script",
        )
        negative = (
            "Fix my Wi-Fi",
            "Create a shopping list",
            "Add this to my memory",
            "Remove that memory",
            "How do I fix this?",
            "Why is this code broken?",
        )

        for message in positive:
            with self.subTest(message=message):
                self.assertEqual(router.match_work(message)["type"], "WORK")

        for message in negative:
            with self.subTest(message=message):
                self.assertIsNone(router.match_work(message))

    def test_online_normal_request_is_cloud(self):
        with patch.object(router, "route_with_groq", side_effect=OSError("Groq unavailable")):
            plan = router.route_online("Explain Python dictionaries")
        self.assertEqual([step["type"] for step in plan["steps"]], ["CLOUD"])

    def test_online_explicit_tool_request(self):
        with patch.object(
            router,
            "route_with_groq",
            return_value={
                "steps": [
                    {"type": "TOOL", "action": "OPEN_APP", "target": "chrome", "content": None}
                ]
            },
        ):
            plan = router.route_online("Open Chrome")
        self.assertEqual(plan["steps"][0]["type"], "TOOL")
        self.assertEqual(plan["steps"][0]["target"], "chrome")

    def test_tool_then_cloud_order(self):
        with patch.object(
            router,
            "route_with_groq",
            return_value={
                "steps": [
                    {"type": "TOOL", "action": "OPEN_APP", "target": "chrome", "content": None},
                    {"type": "CLOUD", "action": None, "target": None, "content": "explain Python dictionaries"},
                ]
            },
        ):
            steps = router.route_online("Open Chrome and explain Python dictionaries")["steps"]
        self.assertEqual([step["type"] for step in steps], ["TOOL", "CLOUD"])

    def test_cloud_then_tool_order(self):
        with patch.object(
            router,
            "route_with_groq",
            return_value={
                "steps": [
                    {"type": "CLOUD", "action": None, "target": None, "content": "Explain Python dictionaries"},
                    {"type": "TOOL", "action": "OPEN_APP", "target": "chrome", "content": None},
                ]
            },
        ):
            steps = router.route_online("Explain Python dictionaries, then open Chrome")["steps"]
        self.assertEqual([step["type"] for step in steps], ["CLOUD", "TOOL"])

    def test_how_question_is_not_a_tool_request(self):
        with patch.object(
            router,
            "route_with_groq",
            return_value={
                "steps": [
                    {"type": "CLOUD", "action": None, "target": None, "content": "How do I open Chrome?"}
                ]
            },
        ):
            steps = router.route_online("How do I open Chrome?")["steps"]
        self.assertEqual(steps[0]["type"], "CLOUD")

    def test_explicit_local_mode_switch_is_deterministic(self):
        with patch.object(router, "route_with_groq", side_effect=AssertionError("switch must not call Groq")):
            plan = router.route_online("Jarvis, go into local mode.")
        self.assertEqual(plan, {"steps": [], "switch_to": "LOCAL"})

    def test_local_memory_request_and_local_reasoning(self):
        self.assertEqual(router.route_local("What GPU do I have?")["steps"][0]["type"], "MEMORY")
        self.assertEqual(router.route_local("Explain Python dictionaries")["steps"][0]["type"], "FALLBACK")

    def test_local_mode_supports_shared_tools(self):
        plan = router.route_local("Open Chrome")
        self.assertEqual(plan["steps"][0]["type"], "TOOL")
        self.assertEqual(plan["steps"][0]["target"], "chrome")

    def test_local_mode_never_routes_to_cloud(self):
        for query in ("What GPU do I have?", "Explain Python dictionaries", "Open Chrome"):
            plan = router.route_local(query)
            self.assertNotIn("CLOUD", [step["type"] for step in plan["steps"]])

    def test_return_to_online_mode(self):
        state = ModeState(Mode.LOCAL)
        state.apply_command("Jarvis, return to online mode")
        self.assertEqual(state.mode, Mode.ONLINE)

    def test_mode_commands_require_exact_phrase(self):
        state = ModeState()
        self.assertIsNone(state.apply_command("Tell me about local mode"))
        self.assertEqual(state.mode, Mode.ONLINE)
        self.assertEqual(state.apply_command("Jarvis, go into local mode."), Mode.LOCAL)

    def test_online_router_never_emits_local_or_memory_steps(self):
        with patch.object(
            router,
            "route_with_groq",
            return_value={
                "steps": [
                    {"type": "LOCAL", "action": None, "target": None, "content": "private"}
                ]
            },
        ):
            plan = router.route_online("What GPU do I have?")
        self.assertEqual([step["type"] for step in plan["steps"]], ["CLOUD"])
        self.assertFalse(any(step.get("memory") for step in plan["steps"]))


if __name__ == "__main__":
    unittest.main()
