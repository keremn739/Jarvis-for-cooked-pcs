import json
import unittest
from unittest.mock import patch

from mode import Mode, ModeState
import router
import main


class ModeArchitectureBenchmark(unittest.TestCase):

    # ---------------------------------------------------------
    # MODE STATE
    # ---------------------------------------------------------

    def test_starts_online(self):
        state = ModeState()
        self.assertEqual(state.mode, Mode.ONLINE)

    def test_explicit_local_switch(self):
        state = ModeState()

        result = state.apply_command("go into local mode")

        self.assertEqual(result, Mode.LOCAL)
        self.assertEqual(state.mode, Mode.LOCAL)

    def test_explicit_online_switch(self):
        state = ModeState()
        state.apply_command("go into local mode")

        result = state.apply_command("go into online mode")

        self.assertEqual(result, Mode.ONLINE)
        self.assertEqual(state.mode, Mode.ONLINE)

    def test_jarvis_prefix_is_allowed(self):
        state = ModeState()

        result = state.apply_command("Jarvis, go into local mode")

        self.assertEqual(result, Mode.LOCAL)

    def test_discussing_local_mode_does_not_switch(self):
        state = ModeState()

        result = state.apply_command(
            "Can you explain what local mode does?"
        )

        self.assertIsNone(result)
        self.assertEqual(state.mode, Mode.ONLINE)

    # ---------------------------------------------------------
    # ONLINE ROUTER
    # ---------------------------------------------------------

    def test_online_tool(self):
        result = router.route_online("Open Chrome")

        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0]["type"], "TOOL")
        self.assertEqual(result["steps"][0]["action"], "OPEN_APP")
        self.assertEqual(result["steps"][0]["target"], "chrome")

    def test_online_cloud(self):
        result = router.route_online("What is a GPU?")

        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0]["type"], "CLOUD")

    def test_online_multi_intent_tool_then_cloud(self):
        result = router.route_online(
            "Open Chrome and explain Python dictionaries."
        )

        self.assertEqual(
            [step["type"] for step in result["steps"]],
            ["TOOL", "CLOUD"],
        )

    def test_online_multi_intent_cloud_then_tool(self):
        result = router.route_online(
            "Explain Python dictionaries, then open Chrome."
        )

        self.assertEqual(
            [step["type"] for step in result["steps"]],
            ["CLOUD", "TOOL"],
        )

    def test_online_cannot_emit_local_memory(self):
        result = router.route_online("What GPU do I have?")

        for step in result["steps"]:
            self.assertNotEqual(step["type"], "LOCAL")
            self.assertNotEqual(step["type"], "MEMORY")
            self.assertFalse(step.get("memory", False))

    def test_online_explicit_local_switch(self):
        result = router.route_online("Go into local mode")

        self.assertEqual(result["steps"], [])
        self.assertEqual(result["switch_to"], "LOCAL")

    # ---------------------------------------------------------
    # LOCAL ROUTER
    # ---------------------------------------------------------

    def test_local_tool(self):
        result = router.route_local("Open Chrome")

        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0]["type"], "TOOL")
        self.assertEqual(result["steps"][0]["action"], "OPEN_APP")
        self.assertEqual(result["steps"][0]["target"], "chrome")

    def test_local_memory(self):
        result = router.route_local("What GPU do I have?")

        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0]["type"], "MEMORY")

    def test_local_fallback(self):
        result = router.route_local(
            "Explain Python dictionaries."
        )

        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0]["type"], "FALLBACK")

    def test_local_never_cloud(self):
        test_messages = [
            "What is a GPU?",
            "Explain Python dictionaries.",
            "What is Python?",
            "Tell me about computer engineering.",
        ]

        for message in test_messages:
            result = router.route_local(message)

            for step in result["steps"]:
                self.assertNotEqual(
                    step["type"],
                    "CLOUD",
                    msg=f"LOCAL mode emitted CLOUD for: {message}",
                )

    def test_local_explicit_online_switch(self):
        result = router.route_local("Go into online mode")

        self.assertEqual(result["steps"], [])
        self.assertEqual(result["switch_to"], "ONLINE")

    # ---------------------------------------------------------
    # LOCAL MODE TOOL + MEMORY + FALLBACK SEPARATION
    # ---------------------------------------------------------

    def test_local_route_types_are_restricted(self):
        test_messages = [
            "Open Chrome",
            "What GPU do I have?",
            "Explain Python dictionaries.",
        ]

        allowed = {"TOOL", "MEMORY", "FALLBACK"}

        for message in test_messages:
            result = router.route_local(message)

            for step in result["steps"]:
                self.assertIn(
                    step["type"],
                    allowed,
                    msg=f"Invalid LOCAL route for: {message}",
                )


    # ---------------------------------------------------------
    # NO LEGACY ROUTER
    # ---------------------------------------------------------

    def test_route_message_removed(self):
        self.assertFalse(
            hasattr(router, "route_message"),
            "Legacy route_message() should be removed.",
        )


if __name__ == "__main__":
    unittest.main()
