"""
scripts/debug_ocr.py

Run the TextDetector directly on an image and print detection + OCR results
for debugging why PHI detections are not being produced.

Usage:
python3 scripts/debug_ocr.py --input path/to/page.png
"""
import sys
from pathlib import Path
import json
import logging

# ensure repo on path
try:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("debug_ocr")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    args = p.parse_args()

    img_path = Path(args.input)
    if not img_path.exists():
        log.error("Input not found: %s", img_path)
        sys.exit(2)

    try:
        from src.core.config import SETTINGS
        from src.ocr.text_detector import TextDetector
    except Exception as e:
        log.exception("Import failed: %s", e)
        sys.exit(2)

    td = TextDetector(SETTINGS.ocr)
    # load image using imageio so detector sees same array
    from imageio import imread
    img = imread(str(img_path))

    try:
        dets = td.detect_text_regions(img)
        print(json.dumps({"num_detections": len(dets), "detections": dets}, indent=2, default=str))
    except Exception:
        log.exception("detect_text_regions failed")

    try:
        # For extraction, call extract_text_content on the same detections
        extracted = td.extract_text_content(img, dets)
        print("\n=== extracted OCR content ===\n")
        print(json.dumps(extracted, indent=2, default=str))
    except Exception:
        log.exception("extract_text_content failed")


if __name__ == '__main__':
    main()
