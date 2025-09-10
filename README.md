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

## Phase 1: India-Specific Objectives

This phase focuses on detecting and reliably masking India-relevant Personally Identifiable Information (PII) and related contextual fields. The goal is to prioritize high-risk identifiers while keeping latency and auditability in mind.

### Critical entities to detect (Presidio)
- AADHAAR_NUMBER — Unique national identity number; must be detected with highest priority because it directly identifies individuals.
- PAN_NUMBER — Tax identifier commonly used in financial transactions; leakage poses high financial risk.
- BANK_ACCOUNT_NUMBER — Sensitive financial identifier; detection prevents direct financial exposure.
- CREDIT_CARD — Payment card numbers; immediate risk for fraud if exposed.
- PASSPORT_NUMBER — Government-issued travel ID; high-risk identifier for identity theft.
- DRIVING_LICENSE — Government-issued license; considered a critical identity document in many workflows.

Each critical entity should be configured in the PII detector with strict patterns and validation logic (checksums/format checks where applicable).

### High-priority entities
- PERSON — Personal names; high recall required because names frequently co-occur with other identifiers.
- EMAIL_ADDRESS — Common contact identifier used for account recovery and contact; medium-high sensitivity.
- PHONE_NUMBER — Direct contact number; used for authentication and contact, should be redacted.
- ADDRESS — Postal/home/work addresses; sensitive location information often appears in documents.
- DATE_OF_BIRTH — Key demographic identifier that helps re-identification when combined with other data.
- VOTER_ID — Local identity document often used in administration; treat as high-priority.

High-priority entities should favor recall (catch most instances) while maintaining acceptable precision through contextual rules.

### Contextual entities
- HEALTH_ID — Healthcare-specific identifier (if present) useful for linking records; context-dependent sensitivity.
- EMPLOYEE_ID — Organization-scoped identifier; sensitive within enterprise contexts.
- GSTIN — Business tax identifier; sensitive for corporate privacy and regulatory reasons.
- IP_ADDRESS — Network identifier; useful for tracing and may be sensitive in logs.
- URL — Links that may contain query strings or tokens; treat as contextual PII when they include secrets.

Contextual entities are useful signals for downstream filters and auditing; they may be redacted or flagged depending on policy.

### Emerging entities
- BIOMETRIC_DATA — Fingerprint/face templates or descriptors; highly sensitive and generally should be removed or blocked.
- DEVICE_ID — Mobile/IoT device identifiers; useful for linking user behavior but sensitive.
- MOTHER_NAME — Additional familial identifier used in regional forms; may increase re-identification risk.
- FATHER_NAME — Familial identifier; similar sensitivity to `MOTHER_NAME`.

Emerging entities may require specialized detectors or additional policy review before automated redaction is enabled.

### Success criteria
The project will be considered successful for Phase 1 when the following measurable targets are met:

- Critical entities: Precision ≥ 90% — minimize false positives on high-risk identifiers to avoid unnecessary data loss.
- High-priority & Contextual entities: Recall ≥ 85% — ensure most instances are detected so downstream masking or review can occur.
- Latency: ≤ 300 ms per page (average) — end-to-end processing time per page including OCR and PII detection for interactive workflows.

Meeting these targets requires realistic evaluation datasets, per-entity validation (format checks), and configuration tuning (thresholds, contextual rules, and model selection).

## Local Setup

Follow these steps to create a reproducible local development environment and pin dependency versions.

1) Create and activate a Python 3.11 virtual environment named `.venv`

```bash
# create the venv (uses system python3.11)
python3.11 -m venv .venv
# activate (macOS / Linux / zsh)
source .venv/bin/activate
```

Note: using a dedicated virtual environment keeps project dependencies isolated from your system Python and other projects.

2) Upgrade pip to the latest version

```bash
pip install --upgrade pip
```

Note: newer pip versions handle wheels and dependency resolution better; upgrading avoids many install issues.

3) Install Python dependencies from `requirements.txt`

```bash
pip install -r requirements.txt
```

Note: `requirements.txt` should list pinned versions for reproducibility; if the file does not exist yet, see step 5 to snapshot your environment after installing key packages.

4) Download the spaCy small English model

```bash
python -m spacy download en_core_web_sm
```

Note: spaCy models are downloaded separately from Python packages and are required for the NER-based PHI detection flow.

5) Pin installed package versions back to `requirements.txt`

```bash
# Capture exactly what's installed in the venv so CI and collaborators can reproduce the environment
pip freeze > requirements.txt
```

Note: After you finish installing and verifying everything, overwrite or create `requirements.txt` with `pip freeze` to lock versions. Commit `requirements.txt` to source control for reproducible builds.
