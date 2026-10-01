import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import router
from api import groq_router


def _step(step_type, *, action=None, target=None, content=None):
    step = {
        "type": step_type,
        "action": action,
        "target": target,
        "content": content,
    }
    if step_type == "TOOL":
        step.update(memory=False, reason=None)
    return step


class GroqRouterClientTests(unittest.TestCase):
    def test_calls_groq_with_strict_schema_and_low_reasoning(self):
        plan = {"steps": [_step("CLOUD", content="Explain Python")]}
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(plan))
                )
            ]
        )

        result = groq_router.route_with_groq("Explain Python", client=client)

        self.assertEqual(result, plan)
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "openai/gpt-oss-20b")
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertEqual(kwargs["max_completion_tokens"], 512)
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["response_format"]["type"], "json_schema")
        self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])
        self.assertEqual(
            kwargs["response_format"]["json_schema"]["schema"],
            groq_router.ROUTER_SCHEMA,
        )
        self.assertIn("NEVER ANSWER THE USER", kwargs["messages"][0]["content"])
        self.assertIn('"Open Chrome." = TOOL / OPEN_APP / chrome.', kwargs["messages"][0]["content"])
        self.assertIn('"Run the test suite." must produce a WORK step.', kwargs["messages"][0]["content"])
        variants = groq_router.ROUTER_SCHEMA["properties"]["steps"]["items"]["anyOf"]
        self.assertEqual(len(variants), 3)
        discriminators = [
            variant["properties"]["type"]["enum"][0] for variant in variants
        ]
        self.assertEqual(discriminators, ["TOOL", "WORK", "CLOUD"])
        for variant in variants:
            self.assertFalse(variant["additionalProperties"])
        tool_schema, work_schema, cloud_schema = variants
        self.assertEqual(
            set(tool_schema["required"]),
            {"type", "action", "target", "content", "memory", "reason"},
        )
        self.assertEqual(
            set(work_schema["required"]), {"type", "action", "target", "content"}
        )
        self.assertEqual(
            set(cloud_schema["required"]), {"type", "action", "target", "content"}
        )
        self.assertEqual(
            set(tool_schema["properties"]["action"]["enum"]),
            set(groq_router.TOOL_REGISTRY),
        )
        expected_targets = list(
            dict.fromkeys(
                target
                for definition in groq_router.TOOL_REGISTRY.values()
                for target in definition["targets"]
            )
        )
        self.assertEqual(tool_schema["properties"]["target"]["enum"], expected_targets)
        self.assertEqual(tool_schema["properties"]["memory"]["type"], "boolean")
        self.assertEqual(tool_schema["properties"]["reason"]["type"], ["string", "null"])
        self.assertEqual(tool_schema["properties"]["content"]["type"], ["string", "null"])
        self.assertNotIn("depends_on", json.dumps(groq_router.ROUTER_SCHEMA))
        self.assertNotIn("run_if", json.dumps(groq_router.ROUTER_SCHEMA))
        self.assertNotIn("condition", json.dumps(groq_router.ROUTER_SCHEMA))

    def test_malformed_groq_json_raises_for_safe_router_fallback(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
        )

        with self.assertRaises(json.JSONDecodeError):
            groq_router.route_with_groq("Open Chrome", client=client)


class OnlineGroqRoutingTests(unittest.TestCase):
    def assert_cloud_fallback(self, result, message):
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0]["type"], "CLOUD")
        self.assertEqual(result["steps"][0]["content"], message)

    def _route_with_plan(self, message, steps):
        with patch.object(router, "route_with_groq", return_value={"steps": steps}):
            return router.route_online(message)

    def test_single_tool(self):
        result = self._route_with_plan(
            "Open Chrome", [_step("TOOL", action="OPEN_APP", target="chrome")]
        )
        self.assertEqual(result["steps"][0]["type"], "TOOL")
        self.assertEqual(result["steps"][0]["target"], "chrome")

    def test_open_chrome_with_period_is_supported_tool(self):
        result = self._route_with_plan(
            "Open Chrome.", [_step("TOOL", action="OPEN_APP", target="chrome")]
        )
        self.assertEqual(
            result["steps"][0],
            {
                "type": "TOOL",
                "action": "OPEN_APP",
                "target": "chrome",
                "content": None,
                "memory": False,
                "reason": None,
            },
        )

    def test_single_work(self):
        result = self._route_with_plan(
            "Fix the app code", [_step("WORK", content="Fix the app code")]
        )
        self.assertEqual([step["type"] for step in result["steps"]], ["WORK"])

    def test_run_test_suite_is_work(self):
        result = self._route_with_plan(
            "Run the test suite.", [_step("WORK", content="Run the test suite.")]
        )
        self.assertEqual([step["type"] for step in result["steps"]], ["WORK"])

    def test_single_cloud(self):
        result = self._route_with_plan(
            "Explain Python", [_step("CLOUD", content="Explain Python")]
        )
        self.assertEqual([step["type"] for step in result["steps"]], ["CLOUD"])

    def test_how_to_open_supported_app_is_cloud(self):
        for message in (
            "How do I open Chrome?",
            "Can you explain how to open Chrome?",
            "Tell me how to launch Chrome.",
        ):
            with self.subTest(message=message):
                result = self._route_with_plan(
                    message, [_step("CLOUD", content=message)]
                )
                self.assertEqual([step["type"] for step in result["steps"]], ["CLOUD"])
                self.assertEqual(result["steps"][0]["content"], message)

    def test_cloud_content_preserves_weather_request(self):
        message = "Tell me the weather."
        result = self._route_with_plan(
            message, [_step("CLOUD", content="Tell me the weather")]
        )
        self.assertEqual([step["type"] for step in result["steps"]], ["CLOUD"])
        self.assertEqual(result["steps"][0]["content"], "Tell me the weather")
        self.assertNotIn("sunny", result["steps"][0]["content"].lower())

    def test_mixed_plan_preserves_weather_intent_and_order(self):
        message = "Tell me the weather, then run the test suite, and finally open Chrome."
        result = self._route_with_plan(
            message,
            [
                _step("CLOUD", content="Tell me the weather"),
                _step("WORK", content="run the test suite"),
                _step("TOOL", action="OPEN_APP", target="chrome"),
            ],
        )
        steps = result["steps"]
        self.assertEqual(
            [step["type"] for step in steps], ["CLOUD", "WORK", "TOOL"]
        )
        self.assertEqual(steps[0]["content"], "Tell me the weather")
        self.assertNotIn("sunny", steps[0]["content"].lower())
        self.assertEqual(steps[1]["content"], "run the test suite")
        self.assertEqual((steps[2]["action"], steps[2]["target"]), ("OPEN_APP", "chrome"))

    def test_fabricated_cloud_answer_rejects_entire_plan(self):
        message = "Tell me the weather, then run the test suite."
        result = self._route_with_plan(
            message,
            [
                _step(
                    "CLOUD",
                    content="The current weather is sunny with a temperature of 75°F.",
                ),
                _step("WORK", content="run the test suite"),
            ],
        )
        self.assert_cloud_fallback(result, message)

    def test_tool_then_work(self):
        result = self._route_with_plan(
            "Open Chrome, then fix the project",
            [
                _step("TOOL", action="OPEN_APP", target="chrome"),
                _step("WORK", content="fix the project"),
            ],
        )
        self.assertEqual([step["type"] for step in result["steps"]], ["TOOL", "WORK"])

    def test_work_then_tool(self):
        result = self._route_with_plan(
            "Fix the project, then open Chrome",
            [
                _step("WORK", content="Fix the project"),
                _step("TOOL", action="OPEN_APP", target="chrome"),
            ],
        )
        self.assertEqual([step["type"] for step in result["steps"]], ["WORK", "TOOL"])

    def test_tool_then_tool(self):
        result = self._route_with_plan(
            "Open Chrome, then Notepad",
            [
                _step("TOOL", action="OPEN_APP", target="chrome"),
                _step("TOOL", action="OPEN_APP", target="notepad"),
            ],
        )
        self.assertEqual([step["target"] for step in result["steps"]], ["chrome", "notepad"])

    def test_cloud_tool_work_order(self):
        result = self._route_with_plan(
            "Tell me the weather, open Chrome, then run the tests",
            [
                _step("CLOUD", content="Tell me the weather"),
                _step("TOOL", action="OPEN_APP", target="chrome"),
                _step("WORK", content="run the tests"),
            ],
        )
        self.assertEqual(
            [step["type"] for step in result["steps"]], ["CLOUD", "TOOL", "WORK"]
        )

    def test_invalid_tool_falls_back_to_cloud(self):
        message = "Open Chrome, then execute an unsupported action"
        result = self._route_with_plan(
            message, [_step("TOOL", action="DELETE_FILE", target=None)]
        )
        self.assert_cloud_fallback(result, message)

    def test_invalid_target_falls_back_to_cloud(self):
        message = "Open Spotify"
        result = self._route_with_plan(
            message, [_step("TOOL", action="OPEN_APP", target="spotify")]
        )
        self.assert_cloud_fallback(result, message)

    def test_empty_steps_fall_back_to_cloud(self):
        message = "Explain Python dictionaries"
        result = self._route_with_plan(message, [])
        self.assert_cloud_fallback(result, message)

    def test_fabricated_work_content_falls_back_to_cloud(self):
        message = "Run the test suite."
        result = self._route_with_plan(
            message, [_step("WORK", content="The tests all passed successfully")]
        )
        self.assert_cloud_fallback(result, message)

    def test_missing_text_content_falls_back_to_cloud(self):
        message = "Tell me the weather."
        result = self._route_with_plan(message, [_step("CLOUD")])
        self.assert_cloud_fallback(result, message)

    def test_tool_content_must_be_null(self):
        message = "Open Chrome."
        result = self._route_with_plan(
            message,
            [_step("TOOL", action="OPEN_APP", target="chrome", content="open Chrome")],
        )
        self.assert_cloud_fallback(result, message)

    def test_tool_memory_flag_must_be_false(self):
        message = "Open Chrome."
        plan_step = _step("TOOL", action="OPEN_APP", target="chrome")
        plan_step["memory"] = True
        result = self._route_with_plan(message, [plan_step])
        self.assert_cloud_fallback(result, message)

    def test_get_time_requires_null_target(self):
        message = "What time is it?"
        result = self._route_with_plan(
            message, [_step("TOOL", action="GET_TIME", target="chrome")]
        )
        self.assert_cloud_fallback(result, message)

    def test_valid_get_time_tool(self):
        result = self._route_with_plan(
            "What time is it?", [_step("TOOL", action="GET_TIME")]
        )
        self.assertEqual(result["steps"][0]["type"], "TOOL")
        self.assertEqual(result["steps"][0]["action"], "GET_TIME")
        self.assertIsNone(result["steps"][0]["target"])

    def test_invalid_middle_step_rejects_entire_plan(self):
        message = "Open Chrome, then do something unsupported, then run tests"
        result = self._route_with_plan(
            message,
            [
                _step("TOOL", action="OPEN_APP", target="chrome"),
                _step("TOOL", action="OPEN_APP", target="spotify"),
                _step("WORK", content="run tests"),
            ],
        )
        self.assert_cloud_fallback(result, message)

    def test_api_failure_falls_back_to_cloud(self):
        with patch.object(router, "route_with_groq", side_effect=RuntimeError("offline")):
            result = router.route_online("What is the weather?")
        self.assertEqual([step["type"] for step in result["steps"]], ["CLOUD"])
        self.assertEqual(result["steps"][0]["content"], "What is the weather?")

    def test_malformed_plan_falls_back_to_cloud(self):
        message = "Explain this"
        with patch.object(
            router, "route_with_groq", return_value='{"steps":[}'
        ):
            result = router.route_online(message)
        self.assert_cloud_fallback(result, message)

    def test_explicit_local_switch_skips_groq(self):
        with patch.object(router, "route_with_groq") as classify:
            result = router.route_online("Go into local mode")
        classify.assert_not_called()
        self.assertEqual(result, {"steps": [], "switch_to": "LOCAL"})


if __name__ == "__main__":
    unittest.main()
