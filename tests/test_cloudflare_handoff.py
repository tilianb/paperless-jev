"""Run against the pinned, patched paperless-jev source (see README)."""

import copy
import json

import httpx
import pytest

from paperless_jev.classifier import build_request, interpret, plan
from paperless_jev.config import DEFAULTS, Config
from paperless_jev.db import Database
from paperless_jev.jev import JevClient, JevError, classifier, classifier_missing
from paperless_jev.paperless import Metadata, PaperlessClient
from paperless_jev.processor import Processor
from paperless_jev.vault import Vault


@pytest.fixture
def cloudflare(monkeypatch, tmp_path):
    token = tmp_path / 'cloudflare-token'
    token.write_text('test-token-not-a-secret')
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'test-account')
    monkeypatch.setenv('CLOUDFLARE_API_TOKEN_FILE', str(token))
    return {'classifier': 'typesafe', 'typesafe_api_key': '', 'model': 'jev-latest'}


@pytest.mark.asyncio
@pytest.mark.parametrize('wrapped', [False, True])
async def test_cloudflare_tagging_response(cloudflare, wrapped):
    questions = {'tag:17': {'type': 'noul', 'instructions': 'Server hosting?'}}
    answer = {'model': 'jev-1.13.0', 'answers': {'tag:17': {'type': 'noul', 'noul': 0.97}},
              'usage': {'input_tokens': 123}}

    def handler(request):
        assert str(request.url) == 'https://api.cloudflare.com/client/v4/accounts/test-account/ai/run'
        assert request.headers['Authorization'] == 'Bearer test-token-not-a-secret'
        assert json.loads(request.content) == {
            'model': 'typesafe/jev',
            'input': {'state': 'Monthly server invoice', 'questions': questions},
        }
        body = {'success': True, 'result': {'state': 'Completed', 'result': answer}} if wrapped else answer
        return httpx.Response(200, json=body)

    assert classifier_missing(cloudflare) is None
    client, model = classifier(cloudflare)
    client._http._transport = httpx.MockTransport(handler)
    async with client:
        result = await client.ask('Monthly server invoice', questions, model)
    assert model == 'typesafe/jev'
    assert result == answer


@pytest.mark.asyncio
@pytest.mark.parametrize('body', [
    {'success': False, 'errors': [{'message': 'provider failure'}]},
    {'success': True, 'result': {'state': 'Queued'}},
    {'answers': {}},
    {'answers': {'tag:17': {'type': 'noul', 'noul': 1.1}}},
])
async def test_invalid_cloudflare_result_cannot_complete_document(cloudflare, body):
    client, model = classifier(cloudflare)
    client._http._transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    async with client:
        with pytest.raises(JevError):
            await client.ask('Invoice', {'tag:17': {'type': 'noul', 'instructions': 'Server?'}}, model)


def test_missing_cloudflare_token_does_not_fall_back_to_typesafe(cloudflare, monkeypatch):
    monkeypatch.delenv('CLOUDFLARE_API_TOKEN_FILE')
    assert classifier_missing(cloudflare)


@pytest.mark.asyncio
async def test_explicit_local_backend_is_not_redirected(cloudflare):
    def handler(request):
        assert str(request.url) == 'http://ollama:11434/v1/systemone'
        assert 'Authorization' not in request.headers
        return httpx.Response(200, json={'answers': {'tag:17': {'type': 'noul', 'noul': 0.8}}})

    async with JevClient(None, base_url='http://ollama:11434') as client:
        client._http._transport = httpx.MockTransport(handler)
        result = await client.ask('Invoice', {'tag:17': {'type': 'noul', 'instructions': '?'}}, 'clef')
    assert result['answers']['tag:17']['noul'] == 0.8


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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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
