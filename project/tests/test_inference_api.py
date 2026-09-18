import torch
import pytest
from fastapi.testclient import TestClient
from project.config import CATEGORICAL
from project.inference import QueryParser, resolve_checkpoint
from project.train import save_bundle
from project.api import create_app


def test_bundle_roundtrip_and_api(tiny, tmp_path, monkeypatch):
    model, tokenizer, config, labels = tiny
    # One real optimizer update ensures the fixture bundle exercises fine-tuned weights.
    optimizer = torch.optim.AdamW(model.parameters())
    sum(v.sum() for v in model.parameters()).backward()
    optimizer.step()
    parser = QueryParser(model, tokenizer, config, labels)
    before = parser.predict(['người áo trắng'])
    path = tmp_path / 'bundle'
    save_bundle(path, model, tokenizer, config, labels, {})
    loaded = QueryParser.from_checkpoint(path)
    assert before == loaded.predict(['người áo trắng'])
    (tmp_path / 'best.txt').write_text('bundle\n')
    assert QueryParser.from_checkpoint(tmp_path).predict(['người áo trắng']) == before
    monkeypatch.setenv('PERSON_QUERY_CHECKPOINT', str(tmp_path))
    with TestClient(create_app()) as client:
        assert client.get('/health').status_code == 200
        response = client.post('/parse_query', json={'text': 'người áo trắng'})
        assert response.status_code == 200
        assert response.json() == loaded.parse('người áo trắng')
        for body in ({'text': ''}, {'text': ' '}, {'text': 4}, {'text': 'x', 'extra': 1}):
            result = client.post('/parse_query', json=body)
            assert result.status_code == 422
            assert 'error' in result.json()
        assert client.post('/parse_query', json={'text': 'người . ' * 40}).status_code == 422


def test_checkpoint_resolution_errors_and_absolute_reference(tmp_path):
    with pytest.raises(FileNotFoundError, match='No checkpoint'):
        resolve_checkpoint(tmp_path)
    pointer = tmp_path / 'best.txt'
    pointer.write_text('')
    with pytest.raises(ValueError, match='empty'):
        resolve_checkpoint(tmp_path)
    pointer.write_text('missing-epoch')
    with pytest.raises(FileNotFoundError, match='unavailable'):
        resolve_checkpoint(tmp_path)
    bundle = tmp_path / 'epoch-1'
    bundle.mkdir()
    (bundle / 'manifest.json').write_text('{}')
    pointer.write_text(str(bundle.resolve()))
    assert resolve_checkpoint(tmp_path) == bundle.resolve()


def test_missing_model_returns_503(monkeypatch):
    monkeypatch.delenv('PERSON_QUERY_CHECKPOINT', raising=False)
    with TestClient(create_app()) as client:
        assert client.get('/health').status_code == 503
        assert client.post('/parse_query', json={'text': 'người'}).status_code == 503


def test_unknowns_omitted_and_multiple_accessories(tiny):
    model, tokenizer, config, labels = tiny
    with torch.no_grad():
        for key, head in model.heads.items():
            head.weight.zero_()
            head.bias.fill_(-20)
            if key in CATEGORICAL:
                head.bias[labels[key].index('unknown')] = 20
        model.heads['accessories'].bias[0] = 20
        model.heads['accessories'].bias[2] = 20
    result = QueryParser(model, tokenizer, config, labels).parse('người')
    assert result['hair'] == {'color': 'unknown'}
    assert result['accessories'] == [{'type': 'bag'}, {'type': 'glasses'}]
