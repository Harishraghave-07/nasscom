"""
scripts/run_on_image.py

Simple runner to invoke ClinicalImageMaskingPipeline on a single image file
and print/save the JSON report. Use this when you already have a PNG image
(e.g. converted PDF page) and want to run the pipeline stages starting from
preprocessing.

Usage:
python3 scripts/run_on_image.py --input debug_output/verification/.../page_001_masked.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ensure repo root on sys.path
try:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("run_on_image")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to an image file (PNG/JPG)")
    parser.add_argument("--out-json", help="Optional output JSON path")
    parser.add_argument("--low-ocr", action="store_true", help="Temporarily lower OCR confidence threshold for debugging")
    args = parser.parse_args()

    img_path = Path(args.input)
    if not img_path.exists():
        log.error("Input image not found: %s", img_path)
        sys.exit(2)

    try:
        from src.core.config import SETTINGS
        from src.core.pipeline import ClinicalImageMaskingPipeline
    except Exception as e:
        log.exception("Failed to import app config or pipeline: %s", e)
        sys.exit(2)

    pipeline = ClinicalImageMaskingPipeline(SETTINGS)
    if args.low_ocr:
        try:
            # temporarily lower OCR confidence threshold for this run
            SETTINGS.ocr.confidence_threshold = 0.1
            log.info("Temporarily lowered OCR confidence threshold to 0.1 for debugging")
        except Exception:
            log.exception("Failed to override OCR confidence threshold")

    out_json = Path(args.out_json) if args.out_json else img_path.parent / (img_path.stem + "_run_report.json")

    try:
        report = pipeline.process_single_image(str(img_path), output_path=None, metadata={"source": "manual_image_run"})
        log.info("Report: %s", report)
        out_json.write_text(json.dumps(report, indent=2, default=str))
        print("Wrote report to", out_json)
    except Exception as e:
        log.exception("Processing failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
