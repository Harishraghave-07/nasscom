#!/usr/bin/env python3
"""Visual debugging of PDF->image->OCR->PHI->mask mapping and masking application.

Saves per-page debug artifacts:
 - debug_page_001_raw.png
 - debug_page_001_detections.png  (red boxes + labels)
 - debug_page_001_masked_simulation.png (black boxes applied)
 - debug_page_001_mask_overlay.png (mask visualization)

Usage:
  python3 scripts/debug_masking_coordinates.py --input "Highlighted PHI Details Emily_Dawson.pdf"
  python3 scripts/debug_masking_coordinates.py --input "..." --force-blackbox

The script is defensive if heavy deps (EasyOCR, cv2) are missing.
"""

from __future__ import annotations

import os
import sys
import json
import tempfile
import textwrap
from pathlib import Path
from typing import List, Dict, Tuple

# Make repo importable when running as a script from tools
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import fitz
except Exception:
    fitz = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None

from src.core.config import SETTINGS
from src.ocr.text_detector import TextDetector
from src.integration.presidio_wrapper import get_analyzer
from src.masking.image_inpainter import ImageInpainter


def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 150) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs: List[Path] = []
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed")
    doc = fitz.open(str(pdf_path))
    for i in range(doc.page_count):
        page = doc.load_page(i)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        outp = out_dir / f"debug_page_{i+1:03d}_raw.png"
        pix.save(str(outp))
        imgs.append(outp)
    doc.close()
    return imgs


def load_image_numpy(path: Path) -> np.ndarray:
    if cv2 is not None:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"cv2 failed to load {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        if Image is None:
            raise RuntimeError("No image backend available (cv2 or PIL required)")
        img = Image.open(str(path)).convert("RGB")
        return np.array(img)


def save_image_from_numpy(arr: np.ndarray, path: Path) -> None:
    if Image is not None:
        img = Image.fromarray(arr.astype('uint8'))
        img.save(str(path))
    elif cv2 is not None:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
    else:
        raise RuntimeError("No image backend available to save images")


def draw_detections(image: np.ndarray, detections: List[Dict], out_path: Path) -> None:
    # detections: list of dict with bbox,text,confidence,entity
    if Image is None:
        # fallback: use cv2 to draw
        if cv2 is None:
            raise RuntimeError("No drawing backend available")
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for d in detections:
            x1, y1, x2, y2 = map(int, d.get('bbox', (0,0,0,0)))
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{d.get('entity_type','')[:10]} {d.get('confidence',0):.2f}"
            cv2.putText(bgr, label, (x1, max(10, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.imwrite(str(out_path), bgr)
        return

    img = Image.fromarray(image.astype('uint8')).convert('RGB')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for d in detections:
        x1, y1, x2, y2 = map(int, d.get('bbox', (0,0,0,0)))
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        label = f"{d.get('entity_type','')[:12]} {d.get('confidence',0):.2f}"
        if font:
            draw.text((x1, max(0, y1-12)), label, fill=(255,0,0), font=font)
        else:
            draw.text((x1, max(0, y1-12)), label, fill=(255,0,0))
    img.save(str(out_path))


def apply_blackbox(image: np.ndarray, regions: List[Dict]) -> np.ndarray:
    out = image.copy()
    for r in regions:
        x1, y1, x2, y2 = map(int, r.get('bbox', (0,0,0,0)))
        out[y1:y2, x1:x2] = 0
    return out


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Debug masking coordinates and visualize detections")
    p.add_argument("--input", required=True, help="PDF file to analyze")
    p.add_argument("--out", default="debug_output/masking_debug", help="Output debug directory")
    p.add_argument("--dpi", type=int, default=150, help="Render DPI for PDF->PNG")
    p.add_argument("--force-blackbox", action="store_true", help="Simulate blackbox masking instead of inpainting")
    args = p.parse_args()

    pdf_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"PDF: {pdf_path}")
    print(f"Output debug dir: {out_dir}")

    # Step 1: PDF -> images
    tmpdir = out_dir / "pages"
    tmpdir.mkdir(parents=True, exist_ok=True)
    images = pdf_to_images(pdf_path, tmpdir, dpi=args.dpi)
    print(f"Rendered {len(images)} pages")

    # Initialize components
    td = TextDetector(SETTINGS.ocr)
    analyzer = get_analyzer()
    inpainter = ImageInpainter(SETTINGS.mask)

    for idx, img_path in enumerate(images):
        page_no = idx + 1
        raw_out = out_dir / f"debug_page_{page_no:03d}_raw.png"
        print(f"Processing page {page_no} image: {img_path}")

        img = load_image_numpy(img_path)
        save_image_from_numpy(img, raw_out)

        # Step 2: detect text regions
        try:
            regions = td.detect_text_regions(img)
        except Exception as e:
            print(f"TextDetector failed: {e}")
            regions = []

        # convert to TextRegion-like dicts
        text_regions = []
        for r in regions:
            tr = {
                "bbox": tuple(r.get("bbox")),
                "confidence": float(r.get("confidence", 0.0)),
                "low_confidence": bool(r.get("low_confidence", False)),
            }
            text_regions.append(tr)

        det_img = out_dir / f"debug_page_{page_no:03d}_detections.png"
        # prepare detection dicts with placeholder entity_type
        dets_for_draw = [{"bbox": tr["bbox"], "entity_type": "TEXT", "confidence": tr["confidence"]} for tr in text_regions]
        draw_detections(img, dets_for_draw, det_img)
        print(f"Wrote detection visualization: {det_img}")

        # Step 3: Extract text content for each region
        extracted = []
        try:
            extracted = td.extract_text_content(img, regions)
        except Exception as e:
            print(f"extract_text_content failed: {e}")
            # fall back: leave text empty
            extracted = []
            for tr in text_regions:
                extracted.append({**tr, "text": "", "ocr_lines": []})

        # Print region text summary
        for i, ex in enumerate(extracted):
            print(f" Region {i}: bbox={ex.get('bbox')} conf={ex.get('confidence'):.3f} text_preview='{(ex.get('text') or '')[:80]}'")

        # Step 4: PHI detection per-region (Presidio wrapper)
        phi_regions: List[Dict] = []
        for ex in extracted:
            txt = ex.get("text") or ""
            dets = []
            try:
                dets = analyzer.detect_phi_in_text(txt, {"page": page_no})
            except Exception as e:
                print(f"Presidio analyze failed for region: {e}")
                dets = []

            if not dets and txt and SETTINGS and SETTINGS.phi and SETTINGS.phi.enable_regex_patterns:
                # fallback: simple regex search using SETTINGS.get_phi_patterns
                pats = SETTINGS.get_phi_patterns() if SETTINGS else {}
                for name, pat in (pats or {}).items():
                    try:
                        import re

                        for m in re.finditer(pat, txt, flags=re.IGNORECASE):
                            dets.append({"start": m.start(), "end": m.end(), "score": 0.9, "entity_type": name, "text": m.group(0)})
                    except Exception:
                        continue

            # Map detections to image coords: prefer word-level boxes when available
            word_boxes = ex.get("word_boxes") or []
            for d in dets:
                ent_text = d.get("text") or (txt[d.get("start", 0) : d.get("end", 0)] if isinstance(d, dict) else txt)
                ent_text_norm = (ent_text or "").strip().lower()

                chosen_box = None
                # Try to find an exact word match in word_boxes
                for wb in word_boxes:
                    if (wb.get("text") or "").strip().lower() == ent_text_norm:
                        chosen_box = wb.get("bbox")
                        break

                # If no exact match, attempt substring match
                if chosen_box is None:
                    for wb in word_boxes:
                        if ent_text_norm and ent_text_norm in (wb.get("text") or "").strip().lower():
                            chosen_box = wb.get("bbox")
                            break

                # Fallback: use entire region bbox
                if chosen_box is None:
                    chosen_box = ex.get("bbox")

                mapped = {
                    "bbox": list(map(int, chosen_box)),
                    "entity_type": d.get("entity_type") or d.get("label") or "PHI",
                    "text": ent_text,
                    "confidence": float(d.get("score", d.get("confidence", 0.0))) if isinstance(d, dict) else 0.0,
                }
                phi_regions.append(mapped)

        print(f"Detected {len(phi_regions)} PHI regions on page {page_no}")
        for r in phi_regions:
            print(f"  {r['entity_type']} '{r['text']}' bbox={r['bbox']} conf={r['confidence']:.3f}")

        # Step 5: Show phi_regions (converted) and create visualizations
        dets_for_draw = [{"bbox": r["bbox"], "entity_type": r.get("entity_type"), "confidence": r.get("confidence", 0.0)} for r in phi_regions]
        if dets_for_draw:
            draw_detections(img, dets_for_draw, out_dir / f"debug_page_{page_no:03d}_phi_detections.png")
            print(f"Wrote PHI detection visualization for page {page_no}")

        # Create mask from regions and save mask overlay
        try:
            mask = inpainter.create_mask_from_regions(img.shape, phi_regions)
            # visualize mask overlay (red overlay)
            overlay = img.copy()
            if mask is not None:
                # mask is single channel 0/255
                alpha = 0.6
                red = np.zeros_like(img)
                red[..., 0] = 255
                # blend where mask>0
                mask_bool = (mask > 0)
                overlay[mask_bool] = (alpha * red[mask_bool] + (1 - alpha) * overlay[mask_bool]).astype(np.uint8)
                save_image_from_numpy(overlay, out_dir / f"debug_page_{page_no:03d}_mask_overlay.png")
                print(f"Wrote mask overlay for page {page_no}")
            else:
                print("Mask creation returned None")
        except Exception as e:
            print(f"create_mask_from_regions failed: {e}")

        # Create simulated masked image (blackbox) and save
        if args.force_blackbox:
            sim_masked = apply_blackbox(img, phi_regions)
            save_image_from_numpy(sim_masked, out_dir / f"debug_page_{page_no:03d}_masked_simulation.png")
            print(f"Wrote simulated blackbox masked image for page {page_no}")
        else:
            # try to call inpainting (may be slow/heavy)
            try:
                mask = inpainter.create_mask_from_regions(img.shape, phi_regions)
                inpainted = inpainter.apply_inpainting(img, mask, method=SETTINGS.mask.inpainting_method)
                save_image_from_numpy(inpainted, out_dir / f"debug_page_{page_no:03d}_masked_simulation.png")
                print(f"Wrote simulated inpainted image for page {page_no}")
            except Exception as e:
                print(f"Inpainting failed or unavailable: {e}; falling back to blackbox simulation")
                sim_masked = apply_blackbox(img, phi_regions)
                save_image_from_numpy(sim_masked, out_dir / f"debug_page_{page_no:03d}_masked_simulation.png")
                print(f"Wrote simulated blackbox masked image for page {page_no}")

    print("Done. Debug artifacts written under:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
