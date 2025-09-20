import os
import io
import json
import pytest


def test_e2e_pdf_processing_gateway(gateway_test_client, mock_fast_analyzer, mock_anonymizer):
    # locate sample PDF
    sample = os.path.join(os.path.dirname(__file__), '..', '..', 'fake_phi_document.pdf')
    if not os.path.exists(sample):
        pytest.skip("Sample PDF not found in repo root")

    with open(sample, 'rb') as fh:
        files = {'file': ('fake_phi_document.pdf', fh, 'application/pdf')}
        resp = gateway_test_client.post('/api/v1/process', files=files)

    assert resp.status_code in (200, 201)
    data = resp.json()
    # basic checks: report contains phi_regions and masking metadata
    assert 'phi_regions' in data or 'report' in data
    # If report exists, expect at least one detection from the mock
    if 'phi_regions' in data:
        assert isinstance(data['phi_regions'], list)
        assert any(r.get('phi_type') == 'PERSON' or r.get('phi_type') == 'CREDIT_CARD' for r in data['phi_regions'])
    else:
        # fallback: inspect report structure
        rpt = data.get('report') or {}
        assert isinstance(rpt, dict)
