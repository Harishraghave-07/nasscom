import os
import pytest
from src.core.config import SETTINGS


def test_anonymize_respects_feature_flag(gateway_test_client, mock_fast_analyzer, mock_anonymizer, monkeypatch):
    # Disable replacement
    monkeypatch.setenv('ENABLE_DUMMY_DATA_REPLACEMENT', 'false')
    SETTINGS.enable_dummy_data_replacement = False

    resp = gateway_test_client.post('/api/v1/anonymize', json={'text': 'Hello Emily Dawson, card 4111 1111 1111 1111'})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get('anonymized_text') == 'Hello Emily Dawson, card 4111 1111 1111 1111'
    assert data.get('mappings', []) == []

    # Enable replacement
    monkeypatch.setenv('ENABLE_DUMMY_DATA_REPLACEMENT', 'true')
    SETTINGS.enable_dummy_data_replacement = True
    resp2 = gateway_test_client.post('/api/v1/anonymize', json={'text': 'Hello Emily Dawson, card 4111 1111 1111 1111'})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert '[REDACTED' in data2.get('anonymized_text', '')
    assert isinstance(data2.get('mappings'), list)
