from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from typing import Any, Dict, List
import logging

from src.utils.observability import (
    RequestIDMiddleware,
    PrometheusMiddleware,
    metrics_response,
    measure_inference,
    audit_event,
    get_logger,
)

logger = get_logger(__name__)
app = FastAPI(title="PHI Detection Service")

# middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(PrometheusMiddleware)


class AnalyzeRequest(BaseModel):
    text: str
    meta: Dict[str, Any] = {}


@app.on_event("startup")
async def startup():
    # lazy import and warm up analyzer
    try:
        from src.integration.presidio_wrapper import get_analyzer
        analyzer = get_analyzer()
        logger.info("PHI analyzer initialized: %s", analyzer)
    except Exception as e:
        logger.warning("PHI analyzer not fully available: %s", e)


@app.post("/analyze")
@measure_inference("presidio_fallback")
async def analyze(req: AnalyzeRequest, request: Request = None):
    try:
        # attach correlation id to principal-like structure for auditing
        principal = {"id": "service", "role": "service_account", "correlation_id": getattr(request.state, 'correlation_id', None)}
        from src.integration.presidio_wrapper import get_analyzer
        analyzer = get_analyzer()
        detections = analyzer.detect_phi_in_text(req.text, meta=req.meta)
        # audit non-PHI event: detection count and types (no text)
        audit_event("phi.detect", principal, {"count": len(detections), "types": list({d.get('entity_type') for d in detections})})
        return detections
    except Exception as e:
        logger.exception("PHI analyze failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    data, ctype = metrics_response()
    return Response(content=data, media_type=ctype)
