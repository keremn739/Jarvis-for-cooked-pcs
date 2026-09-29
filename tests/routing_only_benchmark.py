import unittest

from mode import Mode, ModeState
from router import route_local, route_online


def route_types(plan):
    return [step["type"] for step in plan.get("steps", [])]


def route_summary(plan):
    result = []

    for step in plan.get("steps", []):
        step_type = step.get("type")

        if step_type == "TOOL":
            result.append(
                f"TOOL:{step.get('action')}:{step.get('target')}"
            )
        else:
            result.append(step_type)

    return result


class RoutingOnlyBenchmark(unittest.TestCase):

    # ---------------------------------------------------------
    # MODE
    # ---------------------------------------------------------

    def test_01_default_mode_is_online(self):
        state = ModeState()
        self.assertEqual(state.mode, Mode.ONLINE)

    def test_02_explicit_local_switch(self):
        state = ModeState()
        result = state.apply_command("Go into local mode")

        self.assertEqual(result, Mode.LOCAL)
        self.assertEqual(state.mode, Mode.LOCAL)

    def test_03_explicit_online_switch(self):
        state = ModeState(Mode.LOCAL)
        result = state.apply_command("Go into online mode")

        self.assertEqual(result, Mode.ONLINE)
        self.assertEqual(state.mode, Mode.ONLINE)

    def test_04_jarvis_prefix_local(self):
        state = ModeState()
        result = state.apply_command("Jarvis, go into local mode")

        self.assertEqual(result, Mode.LOCAL)

    def test_05_jarvis_prefix_online(self):
        state = ModeState(Mode.LOCAL)
        result = state.apply_command("Jarvis, go into online mode")

        self.assertEqual(result, Mode.ONLINE)

    def test_06_discussing_local_mode_does_not_switch(self):
        state = ModeState()

        result = state.apply_command(
            "Can you explain what local mode does?"
        )

        self.assertIsNone(result)
        self.assertEqual(state.mode, Mode.ONLINE)

    # ---------------------------------------------------------
    # ONLINE ROUTER
    # ---------------------------------------------------------

    def test_07_online_normal_question_is_cloud(self):
        plan = route_online("What is Python?")

        self.assertEqual(route_types(plan), ["CLOUD"])

    def test_08_online_tool_is_tool(self):
        plan = route_online("Open Chrome")

        self.assertEqual(
            route_summary(plan),
            ["TOOL:OPEN_APP:chrome"]
        )

    def test_09_online_memory_question_never_becomes_memory(self):
        plan = route_online("What GPU do I have?")

        self.assertNotIn("MEMORY", route_types(plan))

        for step in plan["steps"]:
            self.assertFalse(step.get("memory", False))

    def test_10_online_tool_then_cloud_preserves_order(self):
        plan = route_online(
            "Open Chrome and then explain Python dictionaries"
        )

        self.assertEqual(
            route_summary(plan),
            ["TOOL:OPEN_APP:chrome", "CLOUD"]
        )

    def test_11_online_cloud_then_tool_preserves_order(self):
        plan = route_online(
            "Explain Python dictionaries and then open Chrome"
        )

        self.assertEqual(
            route_summary(plan),
            ["CLOUD", "TOOL:OPEN_APP:chrome"]
        )

    def test_12_online_local_switch_is_not_cloud(self):
        plan = route_online("Go into local mode")

        self.assertEqual(plan.get("switch_to"), "LOCAL")
        self.assertEqual(plan.get("steps"), [])

    # ---------------------------------------------------------
    # LOCAL ROUTER
    # ---------------------------------------------------------

    def test_13_local_normal_question_is_fallback(self):
        plan = route_local("What is Python?")

        self.assertEqual(route_types(plan), ["FALLBACK"])

    def test_14_local_memory_question_is_memory(self):
        plan = route_local("What GPU do I have?")

        self.assertEqual(route_types(plan), ["MEMORY"])

    def test_15_local_tool_is_tool(self):
        plan = route_local("Open Chrome")

        self.assertEqual(
            route_summary(plan),
            ["TOOL:OPEN_APP:chrome"]
        )

    def test_16_local_never_produces_cloud(self):
        test_messages = [
            "What is Python?",
            "Explain machine learning",
            "What is the latest AI news?",
            "Open Chrome and explain Python",
        ]

        for message in test_messages:
            plan = route_local(message)

            self.assertNotIn(
                "CLOUD",
                route_types(plan),
                msg=f"Local router produced CLOUD for: {message}"
            )

    def test_17_local_online_switch(self):
        plan = route_local("Go into online mode")

        self.assertEqual(plan.get("switch_to"), "ONLINE")
        self.assertEqual(plan.get("steps"), [])

    def test_18_local_route_types_are_restricted(self):
        allowed = {"MEMORY", "TOOL", "FALLBACK"}

        messages = [
            "What is Python?",
            "What GPU do I have?",
            "Open Chrome",
        ]

        for message in messages:
            plan = route_local(message)

            for step_type in route_types(plan):
                self.assertIn(
                    step_type,
                    allowed,
                    msg=f"Invalid LOCAL route for: {message}"
                )

    # ---------------------------------------------------------
    # ARCHITECTURE BOUNDARY
    # ---------------------------------------------------------

    def test_19_online_has_no_memory_route(self):
        messages = [
            "What GPU do I have?",
            "What do you remember about me?",
            "What is my computer?",
        ]

        for message in messages:
            plan = route_online(message)

            self.assertNotIn(
                "MEMORY",
                route_types(plan),
                msg=f"ONLINE produced MEMORY for: {message}"
            )

            for step in plan["steps"]:
                self.assertFalse(step.get("memory", False))

    def test_20_local_has_no_cloud_route(self):
        plan = route_local(
            "What is the latest information about artificial intelligence?"
        )

        self.assertNotIn("CLOUD", route_types(plan))


if __name__ == "__main__":
    unittest.main(verbosity=2)
