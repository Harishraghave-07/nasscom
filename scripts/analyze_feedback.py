"""analyze_feedback.py

Read reviewer feedback CSV and produce a summary of common false positives
and false negatives to guide improvements (thresholds, custom rules).

CSV format expected (header optional):
    document_id,pii_text,feedback_type,correct_label

Where feedback_type is one of: false_positive, false_negative

Usage:
    python3 scripts/analyze_feedback.py feedback.csv --top 10 --out report.json

This script prints a human-readable summary and can write a JSON report.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_feedback(path: Path) -> Iterable[Dict[str, str]]:
    """Yield rows from the CSV as dicts.

    Handles files with or without headers. Expects the columns in order if no
    header is present: document_id, pii_text, feedback_type, correct_label.
    """
    with path.open("r", encoding="utf-8") as fh:
        # Try DictReader first (handles header). If fields are missing, fall back.
        reader = csv.DictReader(fh)
        # If DictReader detected fieldnames and they include our keys, use it.
        if reader.fieldnames and set(map(str.lower, reader.fieldnames)) >= {
            "document_id",
            "pii_text",
            "feedback_type",
            "correct_label",
        }:
            for row in reader:
                # Normalize keys to lowercase names used by the rest of the script
                yield {
                    "document_id": row.get("document_id") or row.get("Document_ID") or "",
                    "pii_text": row.get("pii_text") or row.get("PII_Text") or "",
                    "feedback_type": (row.get("feedback_type") or row.get("Feedback_Type") or "").lower(),
                    "correct_label": row.get("correct_label") or row.get("Correct_Label") or "",
                }
        else:
            # No usable header — rewind and parse positional columns.
            fh.seek(0)
            pos_reader = csv.reader(fh)
            for r in pos_reader:
                if not r:
                    continue
                # Skip rows that look like a header line
                first = r[0].lower()
                if first in ("document_id", "doc_id", "id"):
                    continue
                # Expect at least 4 columns
                if len(r) < 4:
                    # pad with empty strings
                    r = (r + [""] * 4)[:4]
                doc_id, pii_text, fb_type, correct = r[0], r[1], r[2].lower(), r[3]
                yield {"document_id": doc_id, "pii_text": pii_text, "feedback_type": fb_type, "correct_label": correct}


def summarize(rows: Iterable[Dict[str, str]], top: int = 10) -> Dict[str, List[Dict[str, object]]]:
    """Group feedback by pii_text and feedback_type and return top-N lists.

    Returns a dict with keys 'false_positive' and 'false_negative', each a list of
    dicts: {"pii_text": str, "count": int, "examples": List[Tuple[document_id, correct_label]]}
    """
    counters: Dict[str, Counter] = {"false_positive": Counter(), "false_negative": Counter()}
    examples: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(lambda: {"false_positive": [], "false_negative": []})

    for row in rows:
        fb = (row.get("feedback_type") or "").lower()
        if fb not in ("false_positive", "false_negative"):
            # Ignore unknown feedback types
            continue
        text = (row.get("pii_text") or "").strip()
        if not text:
            continue
        counters[fb][text] += 1
        examples[text][fb].append((row.get("document_id") or "", row.get("correct_label") or ""))

    def build_list(fb_type: str) -> List[Dict[str, object]]:
        out = []
        for text, cnt in counters[fb_type].most_common(top):
            exs = examples[text][fb_type][:3]
            out.append({"pii_text": text, "count": cnt, "examples": exs})
        return out

    return {"false_positive": build_list("false_positive"), "false_negative": build_list("false_negative")}


def print_report(summary: Dict[str, List[Dict[str, object]]]) -> None:
    print("\nTop False Positives:\n--------------------")
    for i, item in enumerate(summary.get("false_positive", []), 1):
        print(f"{i}. {item['pii_text']} — {item['count']} occurrences")
        for doc_id, correct in item.get("examples", []):
            print(f"    example doc: {doc_id}, correct_label: {correct}")

    print("\nTop False Negatives:\n--------------------")
    for i, item in enumerate(summary.get("false_negative", []), 1):
        print(f"{i}. {item['pii_text']} — {item['count']} occurrences")
        for doc_id, correct in item.get("examples", []):
            print(f"    example doc: {doc_id}, correct_label: {correct}")


def write_json(summary: Dict[str, List[Dict[str, object]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize human reviewer feedback for PII masking")
    p.add_argument("input", type=Path, help="Path to feedback CSV file")
    p.add_argument("--top", type=int, default=10, help="Top-N items to include")
    p.add_argument("--out", type=Path, default=None, help="Optional path to write JSON report")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        return 2

    rows = load_feedback(args.input)
    summary = summarize(rows, top=args.top)
    print_report(summary)
    if args.out:
        write_json(summary, args.out)
        print(f"\nWritten JSON report to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
