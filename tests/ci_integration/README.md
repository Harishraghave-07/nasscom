CI Integration Tests

Overview
- Tests under this directory are designed to be CI-friendly and avoid heavy ML dependencies by mocking analyzers and anonymizers.
- They exercise API endpoints (gateway) and pipeline behavior (anonymization, fallbacks, concurrency).

Running locally

1. Install test requirements (use a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
pip install pytest httpx fastapi pytest-asyncio
```

2. Run pytest targeting these tests:

```bash
pytest tests/ci_integration -q
```

CI notes
- Set `ENABLE_DUMMY_DATA_REPLACEMENT`, `ENABLE_ROLE_BASED_AUTH`, and other flags via environment variables in CI to test both code paths.
- The tests patch heavy dependencies; no GPU or large NLP packages required.
- If your CI runs in an isolated environment, ensure `python` is available (the repository assumes Python 3.11+ but tests will run with 3.8+).
