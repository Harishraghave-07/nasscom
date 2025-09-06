# Usage

This document shows how to run the CLI and Gradio UI locally and in Docker.

CLI

- Run preprocessing on a folder:

```bash
. .venv/bin/activate
python -m src.core.preprocess --input data/raw --output data/processed
```

- Run PHI detection on a single image:

```bash
python -m src.core.detect --image tests/data/sample.jpg --output out.json
```

Gradio UI

- Start the app locally (after installing deps):

```bash
. .venv/bin/activate
python -m src.ui.gradio_app.main
```

- Or run in Docker:

```bash
docker build -t nasscom .
docker run -p 7860:7860 nasscom
```

Testing & Validation

- Run unit tests:

```bash
pytest
```

- Run validation suite (lint, security scan, tests):

```bash
make validate
```
