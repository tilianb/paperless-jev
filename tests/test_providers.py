"""Classifier providers selectable under Setup: TypeSafe, Cloudflare Workers AI, Ollama and custom."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from paperless_jev import app as app_module
from paperless_jev.jev import JevError, classifier, classifier_missing

QUESTIONS = {"tag:17": {"type": "noul", "instructions": "Server hosting?"}}
ANSWER = {"model": "jev-1.13.0", "answers": {"tag:17": {"type": "noul", "noul": 0.97}}, "usage": {"input_tokens": 123}}


def cloudflare_cfg(**extra):
    return {"classifier": "cloudflare", "cloudflare_account_id": "acct", "cloudflare_api_token": "cf-token",
            "typesafe_api_key": "", "model": "jev-latest", **extra}


async def ask(cfg, handler):
    client, model = classifier(cfg)
    client._http._transport = httpx.MockTransport(handler)
    async with client:
        return await client.ask("Monthly server invoice", QUESTIONS, model)


@pytest.fixture(autouse=True)
def no_cloudflare_env(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN_FILE", raising=False)


@pytest.mark.parametrize("wrapped", [False, True])
async def test_cloudflare_uses_setup_credentials_and_unwraps_result(wrapped):
    def handler(request):
        assert str(request.url) == "https://api.cloudflare.com/client/v4/accounts/acct/ai/run"
        assert request.headers["Authorization"] == "Bearer cf-token"
        assert json.loads(request.content) == {
            "model": "typesafe/jev", "input": {"state": "Monthly server invoice", "questions": QUESTIONS},
        }
        body = {"success": True, "result": {"state": "Completed", "result": ANSWER}} if wrapped else ANSWER
        return httpx.Response(200, json=body)

    assert classifier_missing(cloudflare_cfg()) is None
    assert await ask(cloudflare_cfg(), handler) == ANSWER


@pytest.mark.parametrize("body", [
    {"success": False, "errors": [{"message": "provider failure"}]},
    {"success": True, "result": {"state": "Queued"}},
    {"answers": {}},
    {"answers": {"tag:17": {"type": "noul", "noul": 1.1}}},
])
async def test_incomplete_cloudflare_result_fails(body):
    with pytest.raises(JevError):
        await ask(cloudflare_cfg(), lambda request: httpx.Response(200, json=body))


async def test_cloudflare_falls_back_to_environment_and_prefers_setup(monkeypatch, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("file-token\n")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "env-acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN_FILE", str(token_file))
    seen = []

    def handler(request):
        seen.append((request.url.path, request.headers["Authorization"]))
        return httpx.Response(200, json=ANSWER)

    await ask(cloudflare_cfg(cloudflare_account_id="", cloudflare_api_token=""), handler)
    await ask(cloudflare_cfg(), handler)
    assert seen == [
        ("/client/v4/accounts/env-acct/ai/run", "Bearer file-token"),
        ("/client/v4/accounts/acct/ai/run", "Bearer cf-token"),
    ]


def test_cloudflare_missing_credentials():
    assert classifier_missing(cloudflare_cfg(cloudflare_account_id="")) == "Keine Cloudflare-Konto-ID konfiguriert"
    assert classifier_missing(cloudflare_cfg(cloudflare_api_token="")) == "Kein Cloudflare-API-Token konfiguriert"


@pytest.mark.parametrize("key", ["", "custom-key"])
async def test_custom_provider_posts_to_its_server(key):
    cfg = {"classifier": "custom", "classifier_url": "https://jev.example/", "classifier_model": "jev-x",
           "classifier_api_key": key}

    def handler(request):
        assert str(request.url) == "https://jev.example/v1/systemone"
        assert request.headers.get("Authorization") == (f"Bearer {key}" if key else None)
        assert json.loads(request.content)["model"] == "jev-x"
        return httpx.Response(200, json=ANSWER)

    assert classifier_missing(cfg) is None
    assert classifier_missing({**cfg, "classifier_url": ""}) == "Server-URL oder Modell fehlt"
    assert (await ask(cfg, handler))["answers"] == ANSWER["answers"]


def test_setup_saves_provider_and_keeps_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "DATA_DIR", tmp_path)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN_FILE", str(tmp_path / "missing-token"))
    form = {"classifier_kind": "cloudflare", "cloudflare_account_id": " acct ", "cloudflare_api_token": "cf-secret"}
    with TestClient(app_module.app) as c:
        assert c.post("/setup/typesafe", data=form).status_code == 200
        c.post("/setup/typesafe", data={**form, "cloudflare_api_token": ""})
        cfg = c.app.state.config.all()
        stored = {r["key"]: r["value"] for r in c.app.state.db.query("SELECT key, value FROM settings")}
        page = c.get("/setup").text
    assert (cfg["classifier"], cfg["cloudflare_account_id"], cfg["cloudflare_api_token"]) == ("cloudflare", "acct", "cf-secret")
    assert "cf-secret" not in stored["cloudflare_api_token"]
    assert 'value="cloudflare" checked' in page
    assert "Token file found" not in page
