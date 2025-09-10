"""Simple JSON-lines audit writer for masking events.

Writes one JSON object per line with the schema:
  {time, entity_type, policy, original_len, masked_len, page, bbox}

Files are written to: debug_output/pdf_audit/{YYYY-MM-DD}.jsonl
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Dict, Any


def _get_audit_file_path() -> Path:
    out_dir = Path("debug_output") / "pdf_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    return out_dir / f"{today}.jsonl"


def log_mask(event: Dict[str, Any]) -> bool:
    """Write a single masking event as one JSON line.

    Required fields in `event` (best-effort):
      - entity_type: str
      - policy: str
      - original_text or original_len
      - masked_text or masked_len
      - page (optional)
      - bbox (optional) e.g. [x1,y1,x2,y2]

    Returns True on success, False on failure.
    """
    try:
        p = _get_audit_file_path()
        rec: Dict[str, Any] = {}
        rec["time"] = datetime.datetime.utcnow().isoformat()
        rec["entity_type"] = event.get("entity_type") or event.get("phi_type") or "unknown"
        rec["policy"] = event.get("policy") or event.get("operator") or "unknown"

        if "original_len" in event:
            rec["original_len"] = int(event.get("original_len", 0))
        else:
            ot = event.get("original_text")
            rec["original_len"] = len(ot) if isinstance(ot, (str, bytes)) else 0

        if "masked_len" in event:
            rec["masked_len"] = int(event.get("masked_len", 0))
        else:
            mt = event.get("masked_text")
            rec["masked_len"] = len(mt) if isinstance(mt, (str, bytes)) else 0

        rec["page"] = event.get("page")
        rec["bbox"] = event.get("bbox")

        # open in append mode and write JSONL
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


    def _get_shadow_audit_path() -> Path:
        out_dir = Path("debug_output") / "shadow_audit"
        out_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.date.today().isoformat()
        return out_dir / f"{today}.jsonl"


    def log_shadow_audit(doc_id: str, legacy: Dict[str, Any], presidio: Dict[str, Any]) -> bool:
        """Write a single JSON-line containing both legacy and presidio detector outputs.

        The record includes a `doc_id` to correlate items.
        """
        try:
            p = _get_shadow_audit_path()
            rec: Dict[str, Any] = {
                "time": datetime.datetime.utcnow().isoformat(),
                "doc_id": doc_id,
                "legacy": legacy,
                "presidio": presidio,
            }
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return True
        except Exception:
            return False


__all__ = ["log_mask"]
