from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
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
from src.utils.observability import RequestIDMiddleware, PrometheusMiddleware, metrics_response, audit_event, get_logger

logger = get_logger(__name__)

app = FastAPI(title="Masking Backend")

# Observability middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(PrometheusMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process")
async def process_file(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    # Ensure only allowed file extensions
    if not file.filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".tiff")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    correlation_id = getattr(request.state, "correlation_id", None)
    principal = {"id": "anonymous", "role": "anonymous", "correlation_id": correlation_id}

    tmpdir = None
    infile_path = None

    try:
        use_in_memory = bool(getattr(SETTINGS.processing, "use_in_memory_temp", False))
    except Exception:
        use_in_memory = False

    # Read incoming bytes once
    body = await file.read()

    # Try in-memory first when requested
    if use_in_memory:
        try:
            import io

            memfile = io.BytesIO(body)
            memfile.name = file.filename
            infile_path = memfile
        except Exception:
            infile_path = None

    # If no in-memory file, write to a secure temp dir
    if infile_path is None:
        tmpdir = tempfile.mkdtemp(prefix="masking_api_")
        infile_path = os.path.join(tmpdir, file.filename)
        try:
            with open(infile_path, "wb") as f:
                f.write(body)
        except Exception:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass
            raise

    # At this point, infile_path is either a BytesIO or a filesystem path
    try:
        try:
            pipeline = ClinicalImageMaskingPipeline(SETTINGS)
        except TypeError:
            pipeline = ClinicalImageMaskingPipeline()

        # Prepare a results directory when using disk-backed temp
        result_dir = None
        if tmpdir:
            result_dir = os.path.join(tmpdir, "output")
            os.makedirs(result_dir, exist_ok=True)

        reports = []
        artifacts = []

        # Determine if input is PDF
        is_pdf = False
        try:
            if isinstance(infile_path, str) and infile_path.lower().endswith('.pdf'):
                is_pdf = True
            elif hasattr(infile_path, 'name') and str(infile_path.name).lower().endswith('.pdf'):
                is_pdf = True
        except Exception:
            is_pdf = False

        if is_pdf:
            try:
                import fitz  # PyMuPDF

                if isinstance(infile_path, str):
                    doc = fitz.open(infile_path)
                else:
                    # bytes buffer
                    doc = fitz.open(stream=infile_path.getvalue(), filetype='pdf')

                page_images = []
                pages_dir = os.path.join(result_dir or tempfile.gettempdir(), "pages")
                os.makedirs(pages_dir, exist_ok=True)
                for i in range(len(doc)):
                    page = doc.load_page(i)
                    pix = page.get_pixmap(dpi=150)
                    img_path = os.path.join(pages_dir, f"page_{i+1:03d}.png")
                    pix.save(img_path)
                    page_images.append(img_path)

                for img_path in page_images:
                    page_out = os.path.join(result_dir or tempfile.gettempdir(), os.path.basename(img_path) + "_out")
                    os.makedirs(page_out, exist_ok=True)
                    try:
                        try:
                            rep = pipeline.process_single_image(img_path, output_path=page_out)
                        except TypeError:
                            rep = pipeline.process_single_image(img_path, page_out)
                        reports.append({"page_image": img_path, "report": rep})
                        # audit non-PHI metadata: processing success per page
                        audit_event("mask.process.page", principal, {"page": os.path.basename(img_path), "status": "ok"})
                    except Exception as e:
                        reports.append({"page_image": img_path, "error": str(e)})
                        audit_event("mask.process.page", principal, {"page": os.path.basename(img_path), "status": "error", "error": str(e)})

                # collect artifacts
                if result_dir:
                    for root, _, files in os.walk(result_dir):
                        for fn in files:
                            artifacts.append(os.path.join(root, fn))

                # audit PDF-level event (no PHI content)
                audit_event("mask.process.pdf", principal, {"pages": len(page_images), "status": "ok"})
                return JSONResponse({"status": "ok", "reports": reports, "artifacts": artifacts})
            except Exception as e:
                audit_event("mask.process.pdf", principal, {"status": "error", "error": str(e)})
                raise HTTPException(status_code=500, detail=f"PDF processing failed: {e}")

        else:
            # Single image file: process directly
            try:
                out_base = None
                if result_dir:
                    out_base = os.path.join(result_dir, os.path.basename(getattr(infile_path, 'name', infile_path)) + "_out")
                    os.makedirs(out_base, exist_ok=True)
                try:
                    rep = pipeline.process_single_image(infile_path, output_path=out_base) if out_base else pipeline.process_single_image(infile_path)
                except TypeError:
                    rep = pipeline.process_single_image(infile_path)
                reports.append({"file": getattr(infile_path, 'name', infile_path), "report": rep})
                if result_dir:
                    for root, _, files in os.walk(result_dir):
                        for fn in files:
                            artifacts.append(os.path.join(root, fn))
                # audit file-level metadata only
                audit_event("mask.process.file", principal, {"file": getattr(infile_path, 'name', infile_path), "status": "ok"})
                return JSONResponse({"status": "ok", "reports": reports, "artifacts": artifacts})
            except Exception as e:
                audit_event("mask.process.file", principal, {"file": getattr(infile_path, 'name', infile_path), "status": "error", "error": str(e)})
                raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        audit_event("mask.process.request", principal, {"status": "error", "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            await file.close()
        except Exception:
            try:
                file.file.close()
            except Exception:
                pass
        # cleanup temp dir if we created one
        try:
            if tmpdir:
                shutil.rmtree(tmpdir)
        except Exception:
            pass


@app.get('/metrics')
async def metrics():
    data, ctype = metrics_response()
    return Response(content=data, media_type=ctype)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
