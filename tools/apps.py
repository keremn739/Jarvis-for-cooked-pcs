import os
import shutil
import subprocess


ALLOWED_APPS = {
    "chrome": {
        "Windows": ["chrome"],
        "Linux": ["google-chrome", "chromium", "chromium-browser"],
    },
    "notepad": {
        "Windows": ["notepad"],
        "Linux": [],
    },
    "calculator": {
        "Windows": ["calc"],
        "Linux": [],
    },
}


def open_app(app_name):
    app = ALLOWED_APPS.get(app_name.lower())

    if app is None:
        return False

    system = "Windows" if os.name == "nt" else "Linux"
    candidates = app.get(system, [])

    for command in candidates:
        executable = shutil.which(command)

        if executable is None:
            continue

        try:
            subprocess.Popen([executable])
            return True
        except OSError:
            continue

    return False
