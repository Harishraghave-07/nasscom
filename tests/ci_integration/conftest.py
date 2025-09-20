import os
import pytest
from fastapi.testclient import TestClient

# Import app factories or entrypoints
try:
    from services.phi_service.app import app as phi_app
except Exception:
    phi_app = None

try:
    from src.api.server import create_app as create_gateway_app
except Exception:
    create_gateway_app = None


@pytest.fixture(scope="session")
def phi_test_client():
    if phi_app is None:
        pytest.skip("PHI service app not importable; ensure service scaffold is present")
    return TestClient(phi_app)


@pytest.fixture(scope="session")
def gateway_test_client():
    if create_gateway_app is None:
        pytest.skip("Gateway app factory not importable; ensure src.api.server.create_app exists")
    app = create_gateway_app()
    return TestClient(app)


@pytest.fixture
def mock_fast_analyzer(monkeypatch):
    """Replace heavy analyzers with a deterministic fast mock."""
    class MockAnalyzer:
        def detect_phi_in_text(self, text, meta=None):
            # Very simple rule-based detections for CI: look for common patterns
            dets = []
            if "Emily Dawson" in text:
                dets.append({"start": text.find("Emily Dawson"), "end": text.find("Emily Dawson") + len("Emily Dawson"), "text": "Emily Dawson", "phi_type": "PERSON", "confidence": 0.99})
            if "4111 1111 1111 1111" in text:
                idx = text.find("4111 1111 1111 1111")
                dets.append({"start": idx, "end": idx + len("4111 1111 1111 1111"), "text": "4111 1111 1111 1111", "phi_type": "CREDIT_CARD", "confidence": 0.95})
            return dets

    # monkeypatch get_analyzer to return MockAnalyzer
    try:
        from src.api.detector import get_analyzer
        monkeypatch.setattr('src.api.detector.get_analyzer', lambda: MockAnalyzer())
    except Exception:
        # best-effort: patch central presidio wrapper if present
        try:
            import src.integration.presidio_wrapper as pw
            monkeypatch.setattr(pw, 'get_analyzer', lambda: MockAnalyzer())
        except Exception:
            pass
    return MockAnalyzer()


@pytest.fixture
def mock_anonymizer(monkeypatch):
    """Mock the presidio AnonymizerEngine to perform deterministic replacements."""
    class MockAnonymizer:
        def anonymize(self, text, analyzer_results=None, operators=None):
            t = text
            for det in (analyzer_results or []):
                s = int(det.get('start', 0))
                e = int(det.get('end', 0))
                # replace with [REDACTED_<TYPE>]
                phi_type = (det.get('phi_type') or det.get('entity_type') or 'PHI').upper()
                replacement = f"[REDACTED_{phi_type}]"
                t = t[:s] + replacement + t[e:]
            return type('R', (), {'text': t})()

    # patch presidio_anonymizer.AnonymizerEngine where used
    monkeypatch.setitem(os.environ, 'ENABLE_DUMMY_DATA_REPLACEMENT', 'true')
    try:
        import presidio_anonymizer as pa
        monkeypatch.setattr(pa, 'AnonymizerEngine', MockAnonymizer)
    except Exception:
        # patch local imports by monkeypatching the name where imported
        try:
            import src.api.detector as det
            monkeypatch.setattr(det, 'AnonymizerEngine', MockAnonymizer)
        except Exception:
            pass
    return MockAnonymizer()
