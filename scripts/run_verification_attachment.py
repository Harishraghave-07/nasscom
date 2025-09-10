"""
scripts/run_verification_attachment.py

Run the ClinicalImageMaskingPipeline on a provided PDF file (attachment).
- Tries to convert PDF pages to images using PyMuPDF (fitz) or pdf2image.
- Falls back to passing the PDF directly to the pipeline (best-effort).
- Saves masked page images and a combined JSON report to `debug_output/verification/<name>/`.

Usage:
python3 scripts/run_verification_attachment.py \
    --input "/path/to/Highlighted PHI Details Emily_Dawson.pdf"

If no --input is provided, the script will default to the attachment path used in your workspace.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure the repository root is on sys.path so `import src...` works when the
# script is invoked directly (python3 scripts/run_verification_attachment.py).
# This avoids ModuleNotFoundError: No module named 'src' when the package
# isn't installed into the environment.
try:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
except Exception:
    # best-effort; if this fails, imports may still fail and the script will
    # report the original ModuleNotFoundError to help debugging.
    pass

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("run_verification_attachment")

# Default attachment path (from your workspace attachments)
DEFAULT_ATTACHMENT = "/Users/harishraghave/Library/Containers/net.whatsapp.WhatsApp/Data/tmp/documents/924879A7-2C9F-46F9-9DD0-0727F2C4B9B4/Highlighted PHI Details Emily_Dawson.pdf"


def convert_pdf_with_fitz(pdf_path: str, out_dir: str) -> list:
    """Use PyMuPDF (fitz) to render pages to PNG files. Returns list of file paths."""
    try:
        import fitz  # PyMuPDF
        imgs = []
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # render at 2x for better OCR
            out = Path(out_dir) / f"page_{i+1:03d}.png"
            pix.save(str(out))
            imgs.append(str(out))
        return imgs
    except Exception as e:
        log.debug("fitz conversion failed: %s", e)
        return []


def convert_pdf_with_pdf2image(pdf_path: str, out_dir: str) -> list:
    """Use pdf2image to convert pages to images. Returns list of file paths."""
    try:
        from pdf2image import convert_from_path
        imgs = []
        pages = convert_from_path(pdf_path, dpi=200)
        for i, p in enumerate(pages):
            out = Path(out_dir) / f"page_{i+1:03d}.png"
            p.save(out, "PNG")
            imgs.append(str(out))
        return imgs
    except Exception as e:
        log.debug("pdf2image conversion failed: %s", e)
        return []


def try_direct_imageio(pdf_path: str, out_dir: str) -> list:
    """Attempt to read the PDF with imageio.imread (best-effort). Returns list of images saved as files."""
    try:
        import imageio
        arr = imageio.imread(pdf_path)
        # If a single array, write it out; if multiple frames, iterate.
        imgs = []
        if hasattr(arr, "shape"):
            out = Path(out_dir) / "page_001.png"
            import numpy as np
            from imageio import imwrite
            imwrite(str(out), arr)
            imgs.append(str(out))
        else:
            # iterate frames
            for i, frame in enumerate(arr):
                out = Path(out_dir) / f"page_{i+1:03d}.png"
                from imageio import imwrite
                imwrite(str(out), frame)
                imgs.append(str(out))
        return imgs
    except Exception as e:
        log.debug("imageio direct read failed: %s", e)
        return []


def process_images_with_pipeline(image_paths: list, pipeline, out_dir: Path) -> list:
    reports = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in image_paths:
        try:
            stem = Path(img_path).stem
            out_file = str(out_dir / f"{stem}_masked.png")
            log.info("Processing page image %s", img_path)
            rep = pipeline.process_single_image(img_path, output_path=out_file, metadata={"source": "pdf_attachment"})
            rep["input"] = img_path
            rep["masked_output"] = out_file
            reports.append(rep)
            log.info("Wrote masked page %s", out_file)
        except Exception:
            log.exception("Failed to process %s", img_path)
            reports.append({"input": img_path, "error": "processing_failed"})
    return reports


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_ATTACHMENT)
    p.add_argument("--out", default="debug_output/verification/attachment_run")
    args = p.parse_args()

    input_pdf = Path(args.input)
    if not input_pdf.exists():
        log.error("Input file not found: %s", input_pdf)
        sys.exit(2)

    # load app config and pipeline
    try:
        from src.core.config import SETTINGS
        from src.core.pipeline import ClinicalImageMaskingPipeline
    except Exception as e:
        log.exception("Failed to import application config or pipeline: %s", e)
        sys.exit(2)

    pipeline = ClinicalImageMaskingPipeline(SETTINGS)

    # prepare a temp dir for image conversion
    temp_dir = Path(tempfile.mkdtemp(prefix="verif_pdf_"))
    log.info("Using temp dir %s for page images", temp_dir)

    try:
        # Try PyMuPDF
        images = convert_pdf_with_fitz(str(input_pdf), str(temp_dir))
        if not images:
            images = convert_pdf_with_pdf2image(str(input_pdf), str(temp_dir))
        if not images:
            images = try_direct_imageio(str(input_pdf), str(temp_dir))

        if not images:
            log.warning("Could not convert PDF to images; attempting to process the PDF path directly with pipeline")
            # Try to run the pipeline directly on the PDF file
            out_base = Path(args.out)
            out_base.mkdir(parents=True, exist_ok=True)
            out_mask = out_base / (input_pdf.stem + "_masked.png")
            rep = pipeline.process_single_image(str(input_pdf), output_path=str(out_mask), metadata={"source": "pdf_attachment"})
            rep["input"] = str(input_pdf)
            rep["masked_output"] = str(out_mask)
            combined = {"reports": [rep], "input": str(input_pdf)}
            out_json = out_base / "report.json"
            out_json.write_text(json.dumps(combined, indent=2))
            log.info("Wrote report to %s", out_json)
            sys.exit(0 if not rep.get("error") else 1)

        # process image pages
        out_base = Path(args.out)
        reports = process_images_with_pipeline(images, pipeline, out_base)

        # write combined report
        combined = {"input_pdf": str(input_pdf), "reports": reports}
        out_json = out_base / "report.json"
        out_json.write_text(json.dumps(combined, indent=2, default=str))
        log.info("Wrote combined report to %s", out_json)

        # print summary to stdout
        succ = sum(1 for r in reports if not r.get("error"))
        log.info("Processing complete: %d/%d pages succeeded", succ, len(reports))

    finally:
        # cleanup temp images
        try:
            shutil.rmtree(str(temp_dir))
            log.debug("Removed temp dir %s", temp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    main()
