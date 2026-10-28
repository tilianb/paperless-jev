"""Handoff from an upstream tool: poll only a queue tag and never touch the tool's status tags."""

import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from paperless_jev import app as app_module
from paperless_jev.classifier import build_request, interpret, plan
from paperless_jev.config import DEFAULTS, Config
from paperless_jev.db import Database
from paperless_jev.paperless import Metadata, PaperlessClient
from paperless_jev.processor import Processor
from paperless_jev.vault import Vault

@pytest.mark.parametrize('present', [False, True])
def test_gpt_status_tags_are_never_classified_or_suggested_for_removal(present):
    cfg = copy.deepcopy(DEFAULTS)
    cfg['system_tags'] = ['paperless-gpt-complete']
    cfg['verify_tags'] = False
    for name, spec in cfg['fields'].items():
        spec['enabled'] = name == 'tags'
    meta = Metadata(tags={17: 'hosting', 18: 'paperless-gpt-complete'})
    doc = {'id': 1, 'content': 'Server hosting invoice', 'tags': [18] if present else []}
    req = build_request(doc, meta, {('tag', 18): {'active': not present}}, cfg)
    assert set(req.questions) == {'tag:17'}
    result = interpret({'answers': {'tag:17': {'type': 'noul', 'noul': 0.98}}}, req, meta, cfg)
    assert plan(result, doc, cfg) == {'updates': {'tags_add': [17]}, 'needs_review': False}


@pytest.mark.parametrize('tag_exists', [False, True])
async def test_polling_only_consumes_gpt_handoff_documents(tmp_path, monkeypatch, tag_exists):
    db = Database(tmp_path / 'jev.db')
    cfg = Config(db, Vault(tmp_path))
    cfg.update({'poll_tag': 'jev-pending'})
    cfg.save_instance(None, 'paperless', 'http://paperless', '', 'fake-token', True)
    processor = Processor(db, cfg)

    def handler(request):
        if request.url.path == '/api/tags/':
            rows = [{'id': 18, 'name': 'jev-pending'}] if tag_exists else []
        elif request.url.path == '/api/documents/':
            rows = [{'id': 123 if request.url.params.get('tags__id__all') == '18' else 999}]
        else:
            rows = []
        return httpx.Response(200, json={'results': rows, 'next': None})

    def client(inst):
        pl = PaperlessClient(inst.url, inst.token)
        pl._http._transport = httpx.MockTransport(handler)
        return pl

    monkeypatch.setattr(processor, 'client', client)
    await processor.poll_once()
    assert [row['doc_id'] for row in db.query('SELECT doc_id FROM jobs')] == ([123] if tag_exists else [])


@pytest.mark.parametrize('finished', [False, True])
async def test_handoff_tag_is_removed_without_removing_other_tags(tmp_path, finished):
    db = Database(tmp_path / 'jev.db')
    cfg = Config(db, Vault(tmp_path))
    cfg.update({'poll_tag': 'jev-pending', 'remove_inbox': False})
    iid = cfg.save_instance(None, 'paperless', 'http://paperless', '', 'fake-token', True)
    processor = Processor(db, cfg)
    meta = Metadata(tags={1: 'existing', 2: 'jev-pending', 3: 'ocr-queue',
                          4: cfg.all()['tag_done'], 5: cfg.all()['tag_review'], 6: 'hosting'},
                    inbox_tags={3})
    document = {'id': 123, 'title': 'Keep this title', 'tags': [1, 2, 3]}

    def handler(request):
        document.update(json.loads(request.content))
        return httpx.Response(200, json=document)

    async with PaperlessClient('http://paperless', 'fake-token') as pl:
        pl._http._transport = httpx.MockTransport(handler)
        await processor.apply(cfg.instance(iid), pl, meta, document, {'tags_add': [6]}, cfg.all(), finished)
    assert document['tags'] == [1, 3, 4 if finished else 5, 6]
    assert document['title'] == 'Keep this title'


def test_rules_page_saves_queue_and_system_tags(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "DATA_DIR", tmp_path)
    with TestClient(app_module.app) as c:
        form = {"mode": "dry_run", "poll_tag": " jev-pending ", "system_tags": "gpt-done, ocr-complete,,gpt-done "}
        assert c.post("/rules", data=form).status_code == 200
        cfg = c.app.state.config.all()
        page = c.get("/rules").text
    assert cfg["poll_tag"] == "jev-pending"
    assert cfg["system_tags"] == ["gpt-done", "ocr-complete"]
    assert 'value="gpt-done, ocr-complete"' in page
