"""Neue Korrespondenten: das Sprachmodell schlägt einen Namen vor, Jev prüft ihn.

Angelegt wird erst nach Freigabe (Review) oder mit Auto-Freigabe ab dem Auto-Schwellwert.
"""

from __future__ import annotations

import json
import re
from typing import Any

from . import llm
from .classifier import prepare_text

MAX_NAME = 128
NEW_CRITERION = "A new correspondent named after the sender of this document; it does not exist in the archive yet."

PROMPT = """Who sent or issued this document (company, authority or person - not the recipient)?
Answer only with JSON: {{"name": "<name as printed on the document>"}}. Use {{"name": null}} if unclear.

Document:
{text}
"""


def normalize(name: str) -> str:
    return " ".join(name.split()).casefold()


def existing(names: dict[int, str], name: str) -> int | None:
    """Korrespondent mit gleichem Namen (Gross-/Kleinschreibung und Leerzeichen egal)."""
    key = normalize(name)
    return next((oid for oid, n in names.items() if normalize(n) == key), None)


def clean_name(raw: Any) -> str | None:
    """Nur ein einzeiliger Name passender Länge kommt zu Jev - kein Freitext aus dem Modell."""
    if not isinstance(raw, str) or re.search(r"[\x00-\x1f\x7f]", raw):
        return None
    name = " ".join(raw.split())
    return name if 0 < len(name) <= MAX_NAME else None


async def propose(model: llm.LLM, content: str) -> str | None:
    text = prepare_text(content, 4000)
    try:
        raw = await llm.complete(model, PROMPT.format(text=text), max_tokens=60, temperature=0.0)
    except llm.LLMError:
        return None
    m = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    return clean_name(data.get("name")) if isinstance(data, dict) else None
