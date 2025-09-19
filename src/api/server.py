from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import shutil
import os
from typing import Dict, Any

# Import the pipeline (assumes existing pipeline module)
from src.core.pipeline import ClinicalImageMaskingPipeline
from src.core.config import SETTINGS

app = FastAPI(title="Masking Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process")
async def process_file(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff')):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    tmpdir = tempfile.mkdtemp(prefix="masking_api_")
    try:
        infile_path = os.path.join(tmpdir, file.filename)
        with open(infile_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Pass global SETTINGS if the pipeline expects a config
        try:
            pipeline = ClinicalImageMaskingPipeline(SETTINGS)
        except TypeError:
            pipeline = ClinicalImageMaskingPipeline()

        # Prepare result dir for artifacts
        result_dir = os.path.join(tmpdir, "output")
        os.makedirs(result_dir, exist_ok=True)

        reports = []
        artifacts = []

        # If PDF, render pages to images and process each page image
        if infile_path.lower().endswith('.pdf'):
            try:
                import fitz  # PyMuPDF

                doc = fitz.open(infile_path)
                page_images = []
                pages_dir = os.path.join(result_dir, "pages")
                os.makedirs(pages_dir, exist_ok=True)
                for i in range(len(doc)):
                    page = doc.load_page(i)
                    pix = page.get_pixmap(dpi=150)
                    img_path = os.path.join(pages_dir, f"page_{i+1:03d}.png")
                    pix.save(img_path)
                    page_images.append(img_path)

                # Process each rendered page image with the pipeline
                for img_path in page_images:
                    # create per-page output dir
                    page_out = os.path.join(result_dir, os.path.basename(img_path) + "_out")
                    os.makedirs(page_out, exist_ok=True)
                    try:
                        # prefer keyword arg if available
                        try:
                            rep = pipeline.process_single_image(img_path, output_path=page_out)
                        except TypeError:
                            rep = pipeline.process_single_image(img_path, page_out)
                        reports.append({"page_image": img_path, "report": rep})
                    except Exception as e:
                        reports.append({"page_image": img_path, "error": str(e)})

                # collect artifacts
                for root, _, files in os.walk(result_dir):
                    for fn in files:
                        artifacts.append(os.path.join(root, fn))

                return JSONResponse({"status": "ok", "reports": reports, "artifacts": artifacts})
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"PDF processing failed: {e}")
        else:
            # Single image file: process directly
            try:
                page_out = os.path.join(result_dir, os.path.basename(infile_path) + "_out")
                os.makedirs(page_out, exist_ok=True)
                try:
                    rep = pipeline.process_single_image(infile_path, output_path=page_out)
                except TypeError:
                    rep = pipeline.process_single_image(infile_path)
                reports.append({"file": infile_path, "report": rep})
                for root, _, files in os.walk(result_dir):
                    for fn in files:
                        artifacts.append(os.path.join(root, fn))
                return JSONResponse({"status": "ok", "reports": reports, "artifacts": artifacts})
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            file.file.close()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
