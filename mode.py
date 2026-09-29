"""Small explicit operating-mode state for the interactive session."""

from enum import Enum

class Mode(str, Enum):
    ONLINE = "ONLINE"
    LOCAL = "LOCAL"


class ModeState:
    def __init__(self, mode=Mode.ONLINE):
        self.mode = Mode(mode)

    def switch(self, mode):
        self.mode = Mode(mode)
        return self.mode

    def apply_command(self, message):
        """Apply an exact explicit mode command, returning the new mode or None."""
        text = " ".join(str(message).strip().lower().split())
        text = text.strip(" .!?;,")
        if text.startswith("jarvis,"):
            text = text[len("jarvis,"):].strip()
        if text in {"go into local mode", "enter local mode", "switch to local mode", "activate local mode", "yerel moda geç", "yerel moda gir"}:
            requested = Mode.LOCAL
        elif text in {"go into online mode", "return to online mode", "switch to online mode", "back to online mode", "çevrimiçi moda geç", "normal moda dön"}:
            requested = Mode.ONLINE
        else:
            return None
        if requested == self.mode:
            return None
        return self.switch(requested)
