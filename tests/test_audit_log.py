import os
from pathlib import Path

from src.debug.audit_log import log_mask


def test_log_mask_appends_one_line(tmp_path, monkeypatch):
    # redirect debug_output/pdf_audit to tmp_path
    audit_dir = tmp_path / "pdf_audit"
    monkeypatch.chdir(tmp_path)

    event = {
        "entity_type": "EMAIL",
        "policy": "replace",
        "original_text": "john@example.com",
        "masked_text": "[REDACTED]",
        "page": 1,
        "bbox": [10, 10, 100, 20],
    }
    assert log_mask(event) is True
    # check that file exists and contains exactly one line
    files = list((Path("debug_output") / "pdf_audit").glob("*.jsonl"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
