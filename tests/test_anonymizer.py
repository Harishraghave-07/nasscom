import re
import time
from pathlib import Path

from src.integration.policy_matrix import POLICY_MATRIX
from src.integration.config_loader import get_operator
from src.debug.audit_log import log_mask


SAMPLE_TEXT = "Aadhaar: 1234-5678-9123, email: user@example.com"


def run_anonymizer(text: str):
    """Simple anonymizer harness used for tests.

    - Replaces Aadhaar-like numbers with [AADHAAR]
    - Partially masks emails to first-char + '***' + @domain
    - Logs each detection via `log_mask` using policies from `POLICY_MATRIX`
    Returns (anonymized_text, detections)
    """
    detections = []
    out = text

    # Aadhaar detection: 4-4-4 groups
    aadhaar_re = re.compile(r"\b\d{4}-\d{4}-\d{4}\b")
    for m in aadhaar_re.finditer(text):
        orig = m.group(0)
        masked = "[AADHAAR]"
        detections.append({"entity": "AADHAAR_NUMBER", "original_text": orig, "masked_text": masked})
    out = aadhaar_re.sub("[AADHAAR]", out)

    # Email masking: keep first char, mask rest up to @
    def _mask_email(m):
        first = m.group(1)
        domain = m.group(3)
        return f"{first}***@{domain}"

    email_re = re.compile(r"([A-Za-z0-9._%+-])([A-Za-z0-9._%+-]*?)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
    # collect email detections before substitution
    for m in email_re.finditer(text):
        orig = m.group(0)
        first = m.group(1)
        domain = m.group(3)
        masked = f"{first}***@{domain}"
        detections.append({"entity": "EMAIL_ADDRESS", "original_text": orig, "masked_text": masked})
    out = email_re.sub(_mask_email, out)

    # Log detections using policy mapping
    for d in detections:
        ent = d["entity"]
        op = get_operator(ent, POLICY_MATRIX)
        event = {
            "entity_type": ent,
            "policy": op.get("operator"),
            "original_text": d.get("original_text"),
            "masked_text": d.get("masked_text"),
            "page": 1,
            "bbox": None,
        }
        log_mask(event)

    return out, detections


def test_run_anonymizer_returns_expected_string(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    anonymized, dets = run_anonymizer(SAMPLE_TEXT)
    assert anonymized == "Aadhaar: [AADHAAR], email: u***@example.com"


def test_audit_log_entries_match_policy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    anonymized, dets = run_anonymizer(SAMPLE_TEXT)
    # read today's audit file
    audit_dir = Path("debug_output") / "pdf_audit"
    files = list(audit_dir.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(dets)
    for line, det in zip(lines, dets):
        obj = __import__("json").loads(line)
        ent = det["entity"]
        op = get_operator(ent, POLICY_MATRIX)
        assert obj.get("policy") == op.get("operator")


def test_runtime_under_threshold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t0 = time.perf_counter()
    anonymized, dets = run_anonymizer(SAMPLE_TEXT)
    dur = time.perf_counter() - t0
    assert dur < 0.1
