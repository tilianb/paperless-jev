"""Jeder übersetzbare Text der Oberfläche hat in jeder Sprache eine Fassung mit denselben Platzhaltern und HTML-Tags."""

import re
import string
from pathlib import Path

import httpx
import pytest

from paperless_jev import i18n
from fastapi.testclient import TestClient
from paperless_jev.app import MODE_LABELS, STATUS_LABELS
from paperless_jev.config import FIELD_LABELS
from paperless_jev.llm import PROVIDERS

from .test_processor import client  # noqa: F401
from .test_llm import _patch

SRC = Path(__file__).parent.parent / "src" / "paperless_jev"
CALL = re.compile(r"""\b_\(\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')""")


def _strings() -> set[str]:
    found: set[str] = set()
    for path in [*SRC.glob("templates/*.html"), *SRC.glob("*.py")]:
        for m in CALL.finditer(path.read_text(encoding="utf-8")):
            raw = m.group(1) if m.group(1) is not None else m.group(2)
            found.add(raw.encode().decode("unicode_escape").encode("latin-1").decode("utf-8"))
    found |= {*STATUS_LABELS.values(), *MODE_LABELS.values(), *FIELD_LABELS.values(), *PROVIDERS.values()}
    found |= {"Dokumenttypen", "Korrespondenten", "Tags", "Speicherpfade"}
    return found


def _fields(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


def test_every_ui_string_has_an_english_translation():
    missing = sorted(s for s in _strings() if s not in i18n.EN)
    assert not missing, "Fehlende Übersetzungen:\n" + "\n".join(missing)


def _tags(text: str) -> list[str]:
    return sorted(re.findall(r"</?[a-z]+[^>]*>", text))


def test_placeholders_match():
    for de, en in i18n.EN.items():
        assert _fields(de) == _fields(en), de


def test_every_language_is_complete():
    for code in i18n.LANGUAGES:
        if code in ("de", "en"):
            continue
        tr = i18n.TRANSLATIONS.get(code)
        assert tr is not None, f"locales/{code}.json fehlt"
        assert set(tr) == set(i18n.EN), f"{code}: fehlend {sorted(set(i18n.EN) - set(tr))[:5]}, überzählig {sorted(set(tr) - set(i18n.EN))[:5]}"
        for de, text in tr.items():
            assert text.strip(), (code, de)
            assert _fields(text) == _fields(de), (code, de)
            assert _tags(text) == _tags(i18n.EN[de]), (code, de)


def test_other_languages_translate_and_fall_back_to_english():
    token = i18n.current.set("fr")
    try:
        assert i18n.gettext("Übersicht") == i18n.TRANSLATIONS["fr"]["Übersicht"]
        assert i18n.gettext("gibt es nicht") == "gibt es nicht"
    finally:
        i18n.current.reset(token)


def test_pick_language():
    assert i18n.pick("en", "de-CH") == "en"
    assert i18n.pick(None, "en-US,en;q=0.9") == "en"
    assert i18n.pick(None, "fr-CH,fr;q=0.9") == "fr"
    assert i18n.pick(None, "nl-NL,nl;q=0.9,en;q=0.8") == "nl"
    assert i18n.pick(None, "pt-BR,pt;q=0.9") == "en"
    assert i18n.pick(None, "de-CH,de;q=0.9,en;q=0.8") == "de"
    assert i18n.pick("xx", None) == "en"


@pytest.mark.parametrize("lang, expected", [
    ("en", "Language model returned no usable title"),
    ("de", "Sprachmodell lieferte keinen brauchbaren Titel"),
    ("fr", "Le modèle de langage n’a renvoyé aucun titre utilisable"),
    ("es", "El modelo de lenguaje no devolvió ningún título utilizable"),
    ("it", "Il modello linguistico non ha restituito alcun titolo utilizzabile"),
    ("nl", "Taalmodel heeft geen bruikbare titel teruggegeven"),
    ("pl", "Model językowy nie zwrócił użytecznego tytułu"),
])
def test_title_failure_uses_selected_language(client, monkeypatch, lang, expected):
    client.cookies.set("lang", lang)
    client.app.state.config.update({"title_mode": "llm", "llm_url": "http://model.example", "llm_model": "m"})
    job_id = client.app.state.db.create_job(1, 42, "test")
    _patch(monkeypatch, lambda request: httpx.Response(200, json={"response": ""}))
    for suffix in ("?format=json", ""):
        response = client.post(f"/jobs/{job_id}/title{suffix}")
        assert response.status_code == 200
        message = response.json()["error"] if suffix else response.text
        assert expected in message
        assert "''" not in message


def test_saved_errors_follow_language_when_reopened(client):
    db = client.app.state.db
    job_id = db.create_job(1, 42, "test")
    db.update_job(job_id, status="error", error="Ollama nicht erreichbar: connection refused")
    for lang, expected in (("en", "Ollama unreachable: connection refused"),
                           ("de", "Ollama nicht erreichbar: connection refused")):
        client.cookies.set("lang", lang)
        for path in ("/", "/log", f"/jobs/{job_id}"):
            assert expected in client.get(path).text
    assert db.job(job_id)["error"] == "Ollama nicht erreichbar: connection refused"


def test_connection_failure_uses_selected_language_without_provider_secrets(client, monkeypatch):
    client.cookies.set("lang", "en")
    _patch(monkeypatch, lambda request: httpx.Response(401, text="private-provider-detail"))
    response = client.post("/setup/llm/test", data={"llm_url": "http://model.example", "llm_model": "m"})
    assert "HTTP 401" in response.text
    assert "Sprachmodell" not in response.text
    assert "private-provider-detail" not in response.text


def test_webhook_and_http_errors_use_selected_language(client):
    client.cookies.set("lang", "en")
    assert client.post("/hook/home?token=wrong").json()["detail"] == "Invalid token"
    assert client.get("/jobs/9999").json()["detail"] == "Not found"
    assert client.get("/missing-page").json()["detail"] == "Not found"
    client.cookies.set("lang", "de")
    assert client.get("/missing-page").json()["detail"] == "Nicht gefunden"


@pytest.mark.parametrize("lang, expected", [
    ("en", "Cross-origin request rejected"),
    ("de", "Anfrage einer fremden Seite abgewiesen"),
    ("es", "Solicitud de origen cruzado rechazada"),
    ("fr", "Requête d’origine croisée rejetée"),
    ("it", "Richiesta da un’altra origine rifiutata"),
    ("nl", "Verzoek van een andere oorsprong geweigerd"),
    ("pl", "Odrzucono żądanie z innego źródła"),
])
def test_early_errors_and_request_logs_use_cookie_language(client, caplog, lang, expected):
    client.cookies.set("lang", lang)
    response = client.post("/run", headers={"Origin": "https://other.example"})
    assert response.status_code == 403 and response.text.strip() == expected
    assert expected in caplog.text


def test_validation_errors_retain_codes_with_localized_summary(client):
    client.cookies.set("lang", "de")
    response = client.get("/jobs/not-a-number")
    assert response.status_code == 422
    assert response.json() == {"detail": "Ungültige Anfrage",
                               "errors": [{"loc": ["path", "job_id"], "type": "int_parsing"}]}


def test_unexpected_errors_use_request_language_without_internal_details(client, monkeypatch, caplog):
    async def fail(*args, **kwargs):
        raise ValueError("internal diagnostic")

    monkeypatch.setattr(type(client.app.state.processor), "suggest_title", fail)
    client.app.state.config.update({"title_mode": "llm"})
    with TestClient(client.app, raise_server_exceptions=False) as browser:
        browser.cookies.set("lang", "fr")
        response = browser.post("/jobs/42/title?format=json")
    assert response.status_code == 500
    assert response.json() == {"detail": "Erreur interne du serveur"}
    assert "Erreur interne du serveur" in caplog.text
