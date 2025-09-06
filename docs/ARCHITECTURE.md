# Architecture

This document describes the modules and responsibilities within the `cim` project.

Modules

- preprocessing
  - Purpose: prepare raw images for OCR and downstream detection (deskew, denoise, contrast/brightness adjustments, resizing, format normalization).
  - Inputs: images from `data/raw`.
  - Outputs: cleaned images for OCR in-memory or on-disk (temporary) and metrics about preprocessed frames.
  - Key functions: `preprocess_image(image) -> image`, `batch_preprocess(dir)`, `save_intermediate(image, meta)`.

- OCR
  - Purpose: extract text and positional information (bounding boxes) from images using OCR engines (EasyOCR/Tesseract) and provide confidence scores.
  - Inputs: preprocessed images.
  - Outputs: text blocks with bounding boxes and confidence (e.g., JSON with fields `text`, `bbox`, `confidence`).
  - Key functions: `run_ocr(image) -> List[Token]`, `ocr_to_documents(ocr_results)`.

- PHI detection
  - Purpose: identify PHI within OCR text and image regions. Combines pattern matching (regex), NER models, heuristics, and ML-based classifiers for edge cases.
  - Inputs: OCR tokens, raw image regions, metadata.
  - Outputs: PHI annotations with type, confidence, and bbox mapping for image masking.
  - Key functions: `detect_phi(tokens) -> List[Annotation]`, `classify_region(image_crop) -> Label`.

- masking
  - Purpose: apply deterministic masking/redaction to image regions identified as PHI. Supports options: blur, block, inpainting, or replacement with synthetic tokens.
  - Inputs: image, PHI annotations.
  - Outputs: masked image saved to `data/processed`.
  - Key functions: `apply_mask(image, annotations, strategy)`, `mask_batch()`.

- UI (Gradio)
  - Purpose: enable quick manual review, QA, and batch runs through a web interface.
  - Components: upload panel, preview (side-by-side original/masked), annotation inspector, export masked images, process queue.
  - Entrypoint: `src/ui/gradio_app/main.py`.

Cross-cutting concerns

- Logging and audit: structured logs for each processed item (id, timestamp, operations, user actions) and optional audit trail storage.
- Security & compliance: secure handling of temporary files, access controls for exported data, encryption at rest for processed output when required.
- Tests: unit tests for each module, integration tests for pipeline end-to-end, performance tests for OCR throughput.

Integration points

- Storage: local filesystem or cloud (S3) adapters.
- Models: local or hosted inference endpoints.
- CI/CD: GitHub Actions workflows for linting, testing, security, and container scans.
