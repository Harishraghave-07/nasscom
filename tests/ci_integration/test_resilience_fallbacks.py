import pytest
from unittest.mock import patch


def test_external_phi_service_failure_fallback(gateway_test_client, mock_fast_analyzer, monkeypatch):
    # Patch environment to pretend use_external_phi_service is enabled
    from src.core.config import SETTINGS
    SETTINGS.use_external_phi_service = True

    # Monkeypatch the HTTP client used in pipeline to raise
    try:
        import src.core.pipeline as pipeline

        def raise_on_post(*args, **kwargs):
            raise Exception('external service unreachable')

        monkeypatch.setattr(pipeline, 'httpx', type('X', (), {'post': raise_on_post}))
    except Exception:
        pytest.skip('Pipeline httpx patch failed')

    # Send a detection request that will force pipeline to fallback
    resp = gateway_test_client.post('/api/v1/detect', json={'text': 'Emily Dawson lives here'})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get('detections'), list)
    assert any(d.get('phi_type') == 'PERSON' for d in data.get('detections', []))
