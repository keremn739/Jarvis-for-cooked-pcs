import subprocess


ALLOWED_APPS = {
    "chrome": "chrome",
    "notepad": "notepad",
    "calculator": "calc"
}


def open_app(app_name):
    app = ALLOWED_APPS.get(app_name.lower())

    if app is None:
        return False

    subprocess.Popen(app)
    return True