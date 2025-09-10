"""
scripts/manual_mask_from_image.py

Quick helper to run OCR -> PHI detection -> simple blackbox masking on an image.
This bypasses pipeline filtering to show where PHI would be detected when using
region OCR text for detection.

Usage:
python3 scripts/manual_mask_from_image.py --input path/to/page.png

Saves <input>_manual_masked.png next to the input image.
"""

import sys
from pathlib import Path
import logging
import json

# ensure repo root
try:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("manual_mask")


def draw_blackboxes(image, regions, out_path):
    try:
        import cv2
        img = image.copy()
        for r in regions:
            try:
                x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
            except Exception:
                continue
        cv2.imwrite(str(out_path), img)
        return True
    except Exception:
        # fallback to PIL
        try:
            from PIL import Image, ImageDraw
            img = Image.fromarray(image)
            draw = ImageDraw.Draw(img)
            for r in regions:
                try:
                    x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
                    draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
                except Exception:
                    continue
            img.save(str(out_path))
            return True
        except Exception:
            log.exception("Failed to write masked image")
            return False


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
        from imageio import imread
        img = imread(str(img_path))
    except Exception:
        log.exception("Failed to read image")
        sys.exit(2)

    try:
        from src.core.config import SETTINGS
        from src.ocr.text_detector import TextDetector
        from src.integration.presidio_wrapper import get_analyzer
    except Exception as e:
        log.exception("Import failed: %s", e)
        sys.exit(2)

    td = TextDetector(SETTINGS.ocr)
    regs = td.detect_text_regions(img)
    # perform OCR extract to get text and confidences
    extracted = td.extract_text_content(img, regs)

    # run presidio on region texts if available, else fallback to PHIClassifier
    analyzer = None
    try:
        analyzer = get_analyzer()
    except Exception:
        analyzer = None

    phi_regions = []
    for r in extracted:
        txt = r.get("text", "")
        if not txt.strip():
            continue
        detections = []
        if analyzer is not None:
            try:
                detections = analyzer.detect_phi_in_text(txt, {"source": "manual_mask"})
            except Exception:
                log.exception("Presidio analyzer failed for region")
        else:
            # fallback to PHIClassifier if available
            try:
                from src.phi_detection.phi_classifier import PHIClassifier
                pc = PHIClassifier(SETTINGS.phi)
                detections = pc.detect_phi_in_text(txt, {"source": "manual_mask"})
            except Exception:
                log.exception("Fallback PHI classifier failed")

        if detections:
            # mark bbox as PHI
            out_r = dict(r)
            out_r["detections"] = detections
            phi_regions.append(out_r)

    out_path = img_path.parent / (img_path.stem + "_manual_masked.png")
    if phi_regions:
        # draw blackboxes on phi_regions bboxes
        ok = draw_blackboxes(img, phi_regions, out_path)
        if ok:
            print("Wrote masked image:", out_path)
        else:
            print("Failed to write masked image")
    else:
        print("No PHI detections found in region texts. No masked image generated.")
        # still write an overlay with red boxes of detected regions for inspection
        overlay_path = img_path.parent / (img_path.stem + "_regions_overlay.png")
        try:
            # draw boxes for all regions
            regs_all = [{"bbox": r.get("bbox")} for r in extracted]
            draw_blackboxes(img, regs_all, overlay_path)
            print("Wrote regions overlay:", overlay_path)
        except Exception:
            pass


if __name__ == '__main__':
    main()
