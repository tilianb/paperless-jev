"""Translate diagnostics at display time, including errors saved by older releases."""

import re
from string import Formatter

from .i18n import gettext


# Jobs store canonical German messages, not the language of the worker that ran them.
# Match only app-owned wording; provider responses and HTTP codes remain verbatim.
TEMPLATES = (
    "Sprachmodell lieferte keinen brauchbaren Titel: {detail}",
    "Sprachmodell nicht erreichbar ({url}): {detail}",
    "Sprachmodell HTTP {status}: {detail}",
    "{service} nicht erreichbar: {detail}",
    "{service}: keine Antwort",
)
PATTERNS = [
    (template, re.compile("".join(
        re.escape(literal) + (f"(?P<{field}>.*?)" if field else "")
        for literal, field, _, _ in Formatter().parse(template)
    ), re.DOTALL))
    for template in TEMPLATES
]


def diagnostic(value: str | Exception) -> str:
    text = getattr(value, "public_message", str(value))
    for template, pattern in PATTERNS:
        if match := pattern.fullmatch(text):
            return gettext(template, **match.groupdict())
    return gettext(text)
