# CIM (Clinical Image Masking)

High-level overview

This repository implements a clinical image masking pipeline that detects and masks Protected Health Information (PHI) in images (scanned documents, photos, clinical forms). The project is organized into modular components for preprocessing, OCR, PHI detection, masking, and a Gradio-based UI for manual review and batch processing.

Diagram (placeholder)

[DIAGRAM: data flow]

data/raw -> preprocessing -> OCR -> PHI detection -> masking -> data/processed

Quick links

- Docs: `docs/INDEX.md` (generated)
- Run locally: see `Makefile` target `env` to create a virtualenv and install deps

License: Add your license here

## Quickstart

1. Prepare environment:

    ```bash
    make bootstrap
    ```

2. Create project scaffold (if not already):

    ```bash
    bash create-project.sh
    ```

3. Build and run locally with Docker:

    ```bash
    docker build -t nasscom .
    docker run -p 7860:7860 nasscom
    ```

4. Run tests and validation:

    ```bash
    make validate
    ```
