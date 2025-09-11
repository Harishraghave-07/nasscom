Pinned dependencies and Docker usage
=================================

This project ships a pinned dependency manifest split into core and heavy
optional dependencies to allow reproducible installs while keeping a
lightweight developer install fast.

Files
- `requirements-core.txt` - pinned minimal dependencies required to run the
  code and tests in lightweight environments.
- `requirements-presidio.txt` - heavy / optional dependencies (EasyOCR,
  torch, Presidio, spaCy models). Install these only when you need full
  analyzer/inpainting features or when running E2E tests.
- `Dockerfile` - reproducible container that installs both sets of
  dependencies and runs tests by default.

Local usage
-----------
Create and activate a virtual environment, then install core deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-core.txt
```

If you need the heavy optional deps (Presidio, EasyOCR, torch, spaCy
models), install them separately:

```bash
pip install -r requirements-presidio.txt
```

Docker usage
------------
Build the image (this may take some time due to heavy packages):

```bash
docker build -t nasscom-cim:latest .
```

Run tests inside the container:

```bash
docker run --rm nasscom-cim:latest
```

CI integration
--------------
Use the Dockerfile in a CI workflow for reproducible E2E runs. For PRs,
prefer running the lightweight `requirements-core.txt` tests. Use a
separate manual `workflow_dispatch` job to run the Docker-based E2E tests
with `requirements-presidio.txt`.

Dependency updates
------------------
Consider enabling Dependabot or Renovate to propose updates for both
`requirements-core.txt` and `requirements-presidio.txt`. Keep heavy
dependency upgrades gated behind manual E2E jobs.
