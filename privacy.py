import re


SENSITIVE_PATTERNS = [
    ("password", r"\bpassword\w*"),
    ("passcode", r"\bpasscode\w*"),
    ("şifre", r"\bşifre\w*"),
    ("api_key", r"\bapi key\w*"),
    ("api_anahtarı", r"\bapi anahtarı\w*"),
    ("token", r"\btoken\w*"),
    ("secret", r"\bsecret\w*"),
    ("secret_key", r"\bsecret key\w*"),
    ("private_key", r"\bprivate key\w*"),
    (
        "email",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),

    (
        "phone",
        r"(?<!\d)(?:\+90|0)?[\s-]?5\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"
    ),
]


def find_private_content(message):
    findings = []

    for category, pattern in SENSITIVE_PATTERNS:

        for match in re.finditer(
            pattern,
            message,
            re.IGNORECASE
        ):
            findings.append({
                "category": category,
                "text": match.group(),
                "start": match.start(),
                "end": match.end()
            })

    return findings


def is_private(message):
    return bool(find_private_content(message))