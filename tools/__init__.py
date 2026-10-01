from tools.apps import ALLOWED_APPS, open_app
from tools.system import get_time


TOOL_REGISTRY = {
    "GET_TIME": {
        "handler": get_time,
        "purpose": "get the current local time",
        "trigger": "an explicit request for the current time",
        "targets": (None,),
        "accepts_target": False,
    },
    "OPEN_APP": {
        "handler": open_app,
        "purpose": "open a desktop application",
        "trigger": "an explicit request to open, launch, or start a supported application",
        "targets": tuple(ALLOWED_APPS),
        "accepts_target": True,
    },
}

TOOLS = {action: definition["handler"] for action, definition in TOOL_REGISTRY.items()}


def execute_tool(action, target=None):
    definition = TOOL_REGISTRY.get(action)
    if definition is None:
        return None
    if definition["accepts_target"]:
        return definition["handler"](target)
    return definition["handler"]()
