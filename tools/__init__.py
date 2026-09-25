from tools.system import get_time
from tools.apps import open_app


TOOLS = {
    "GET_TIME": get_time,
    "OPEN_APP": open_app
}


def execute_tool(action, target=None):
    tool = TOOLS.get(action)

    if tool is None:
        return None

    if action == "GET_TIME":
        return tool()

    return tool(target)