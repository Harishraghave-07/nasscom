# Presidio & Legacy Pipeline End-to-End Verification Plan

Purpose: A concise, step-by-step verification plan to exercise the full PII masking service and validate both the legacy (spaCy/regex) and Microsoft Presidio detection + anonymization paths.

Checklist
- [ ] Create a synthetic native-text PDF `verification_test.pdf` containing India-specific PII.
- [ ] Run the pipeline via `scripts/run_verification.py` and capture logs.
- [ ] Inspect the masked PDF visually for correct redaction and no over-masking.
- [ ] Inspect console logs to confirm which pipeline was used.
- [ ] Inspect `debug_output/pdf_audit/` and `debug_output/shadow_audit/` JSONL for expected detections and anonymizer evidence.
- [ ] Produce a short summary conclusion.

---

## 1. Prepare the Test Assets

Action:
- Create a native-text PDF named `verification_test.pdf` and save it under the repo root or `tests/assets/`.

Content to include (exact text lines recommended):
- Patient Name: Rajesh Kumar
- Aadhaar ID: 1234 5678 9012
- PAN: ABCDE1234F
- Contact: +91 9876543210
- The patient reports feeling well.

How to create quickly (Python example):

```python
# scripts/make_verification_pdf.py
from pathlib import Path
from fpdf import FPDF

text_lines = [
    "Patient Name: Rajesh Kumar",
    "Aadhaar ID: 1234 5678 9012",
    "PAN: ABCDE1234F",
    "Contact: +91 9876543210",
    "",
    "The patient reports feeling well.",
]

out = Path("verification_test.pdf")
pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_font("Arial", size=12)
for line in text_lines:
    pdf.cell(0, 8, txt=line, ln=True)
pdf.output(str(out))
print("Wrote:", out)
```

Expected Outcome:
- A native-text PDF `verification_test.pdf` exists and contains the lines above.

---

## 2. Execute the Pipeline and Capture Results

Action:
- Add and run a small launcher script `scripts/run_verification.py` that invokes the `ClinicalImageMaskingPipeline` on `verification_test.pdf` and prints all logs to stdout.

Suggested runner (pseudo-code / example to adapt to your project API):

```python
# scripts/run_verification.py
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

INPUT = Path("verification_test.pdf")
OUTPUT_DIR = Path("debug_output/verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import your pipeline
from src.core.pipeline import ClinicalImageMaskingPipeline

pipeline = ClinicalImageMaskingPipeline(config_path="config/presidio.yaml")
result = pipeline.process_file(str(INPUT), output_dir=str(OUTPUT_DIR))
log.info("Pipeline run result: %s", result)
print("Masked file location:", result.get("masked_pdf"))
```

Action details:
- Ensure `config/presidio.yaml` has `use_presidio: true` or set `presidio_canary_percentage: 100` for a deterministic test.
- Optionally enable `shadow_mode_enabled: true` to keep legacy outputs and compare.
- Run the script and keep the console logs.

Run command:

```bash
python3 scripts/run_verification.py
```

Expected Outcome:
- The script runs without unhandled exceptions.
- A masked PDF (e.g., `verification_test_masked.pdf`) is written under `debug_output/verification/` (or the pipeline's configured output path).
- Audit logs and pipeline logs are generated under `debug_output/` (for example `debug_output/pdf_audit/YYYY-MM-DD.jsonl` and `debug_output/shadow_audit/YYYY-MM-DD.jsonl`).

---

## 3. Verify the Masked Output

Action:
- Open the generated masked PDF (use a PDF viewer or `pdfplumber`/`PyMuPDF` to inspect programmatically).

Expected Outcome:
- Visual confirmation that all PII (Rajesh Kumar, Aadhaar, PAN, phone number) are redacted with black boxes (or the configured redaction style).
- The sentence "The patient reports feeling well." remains unredacted.
- No blanking or cropping of non-PII text.

Programmatic check (optional):
- Use `PyMuPDF`/`pdfplumber` to extract text and assert that the raw PII text is not present in the extracted text:

```python
# optional check
import fitz

doc = fitz.open("debug_output/verification/verification_test_masked.pdf")
text = "\n".join(page.get_text() for page in doc)
assert "Rajesh Kumar" not in text
assert "1234 5678 9012" not in text
assert "ABCDE1234F" not in text
assert "+91 9876543210" not in text
assert "The patient reports feeling well." in text
```

---

## 4. Verify the Detection and Anonymization Logic (Log Analysis)

Action:
- Review the console output produced when you ran `scripts/run_verification.py`.

Expected Outcome:
- Find a log line indicating which PHI detection path was used. Example messages to look for:
  - `PHI classification: used_presidio=True` (or `False`) or similar statements logged by `src/phi_detection/phi_classifier.py`.
- Confirm there are no tracebacks or critical errors.
- Confirm that the pipeline logged detected spans and entities (for example, logger messages enumerating detected entities and their scores).

If using canary:
- If you set `presidio_canary_percentage` < 100, verify whether the run was routed to Presidio or to the legacy path. Re-run until you exercise both paths if desired.

---

## 5. Verify the Audit Trail (Most Important Step)

Action:
- Open the latest file(s) in `debug_output/pdf_audit/` (and `debug_output/shadow_audit/` if `shadow_mode_enabled` is set).
- Search for entries corresponding to `verification_test.pdf` (look for the filename or a generated `doc_id`).

Expected Outcome:
- There are JSONL entries for each detected PII instance: the name, Aadhaar, PAN, and mobile number.
- Each JSON object should include fields similar to:
  - `entity_type` (e.g., `PERSON`, `AADHAAR_NUMBER`, `PAN_NUMBER`, `PHONE_NUMBER`)
  - `start` and `end` offsets (or image bbox coordinates for image-based OCR pipeline)
  - `score` (confidence value)
  - `text` (the original detected text substring)
  - `detector` or `used_presidio` field showing which detector produced the entry

Example minimal JSONL entry (illustrative):

```json
{
  "doc_id": "verification_test.pdf",
  "entity_type": "PERSON",
  "start": 14,
  "end": 26,
  "score": 0.98,
  "text": "Rajesh Kumar",
  "detector": "presidio"
}
```

- Confirm anonymizer evidence: look for either a separate anonymizer log or final string showing anonymized tokens (for example the anonymizer replaced text with `[PERSON]` or similar placeholders), or compare legacy vs presidio outputs in `debug_output/shadow_audit/`.

Notes on offsets vs image coordinates:
- If your pipeline OCRs the PDF pages into words and produces image coordinates (x,y,w,h), ensure the audit contains those boxes. If anonymizer replaced text producing length changes, verify that you also have bounding box info or re-mapped spans for image redaction.

---

## 6. Final Verification Summary

Action:
- Prepare a short written conclusion (1–3 sentences) describing whether the run passed verification and whether audit logs and masked outputs match expectations.

Expected Outcome (example):
- "Verification complete. The PII masking service, with Presidio integration active, detected and masked all target PII (Name, Aadhaar, PAN, Phone) in `verification_test.pdf`. Audit logs in `debug_output/pdf_audit/` contain the corresponding Presidio detections and anonymizer evidence. Non-PII content was not redacted."

---

## Troubleshooting & Common Failure Modes

- No detections found:
  - Confirm `use_presidio: true` or set `presidio_canary_percentage: 100` in `config/presidio.yaml`.
  - Ensure spaCy model(s) and Presidio packages are installed and the `presidio` services are importable.

- Presidio instantiation errors:
  - Check Python package versions. Run `python3 scripts/verify_presidio.py` (exists in the repo) to get a quick status report.

- Audit file missing:
  - Ensure pipeline has write permissions to `debug_output/` and `debug_output/pdf_audit/`.
  - Check pipeline config for `audit` or `debug` toggles.

- Bounding box mismatch after anonymization:
  - If anonymization changes text length, use length-preserving operators (for example masking with block characters or action `redact`) or re-compute boxes from OCR results and apply image-level black boxes.

---

## Optional: Automate the verification run in CI

- Add a GitHub Actions job that:
  - Runs a lightweight smoke test using the `verification_test.pdf` and `scripts/run_verification.py`.
  - Stores `debug_output/` as an artifact for inspection.
- Keep this optional; Presidio may be heavyweight to install in CI. Instead consider mocking Presidio in unit tests and running a single full E2E nightly job on a self-hosted runner that has the dependencies installed.

---

If you want, I can also:
- Create the helper scripts (`scripts/make_verification_pdf.py` and `scripts/run_verification.py`) in the repo now.
- Add a tiny CI job snippet to `.github/workflows/ci.yml` to run this smoke check nightly.

