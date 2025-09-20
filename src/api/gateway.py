from typing import Optional, Dict, Any
import os
import time
from functools import wraps

from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from src.core.config import SETTINGS
from src.api.auth import require_role
import logging

LOG = logging.getLogger("cim.gateway")


# App instance with versioned mounting hints (we keep single app but honor /api/v1/ paths)
app = FastAPI(title="CIM API Gateway")

# Use shared auth.require_role dependency for RBAC and rate-limiting


# Setup CORS from central SETTINGS
cors_origins = getattr(SETTINGS, "cors_origins", []) or []
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# include admin dummy-data router
try:
    from src.api.admin_dummy import router as admin_dummy_router

    app.include_router(admin_dummy_router)
except Exception:
    LOG.exception("Failed to include admin_dummy router; admin endpoints disabled")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/v1/health")
async def health_v1():
    return {"status": "ok", "version": "v1"}


@app.get("/api/v2/health")
async def health_v2():
    return {"status": "ok", "version": "v2"}


def _log_request(request: Request, user: Dict[str, Any]):
    try:
        LOG.info({
            "event": "request",
            "method": request.method,
            "path": request.url.path,
            "user": user.get("sub") if user else None,
            "role": getattr(request.state, "caller_role", None),
            "query": dict(request.query_params),
        })
    except Exception:
        LOG.exception("Failed to log request")


def _log_response(request: Request, status: int, body: Any = None):
    try:
        LOG.info({"event": "response", "path": request.url.path, "status": status})
    except Exception:
        LOG.exception("Failed to log response")


async def _proxy_post(upstream: str, request: Request, user: Dict[str, Any], timeout: float = 30.0):
    # Forward body and headers safely
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    token = os.environ.get("GATEWAY_FORWARD_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(upstream, content=body, headers=headers)
        except httpx.RequestError as e:
            LOG.exception("Upstream request failed: %s", e)
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")
    content_type = resp.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    return JSONResponse(status_code=resp.status_code, content=resp.text)


async def _proxy_get(upstream: str, request: Request, user: Dict[str, Any], timeout: float = 30.0):
    params = dict(request.query_params)
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    token = os.environ.get("GATEWAY_FORWARD_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(upstream, params=params, headers=headers)
        except httpx.RequestError as e:
            LOG.exception("Upstream request failed: %s", e)
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")
    content_type = resp.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    return JSONResponse(status_code=resp.status_code, content=resp.text)


@app.post("/api/v1/route/{service_name}")
async def route_request_v1(request: Request, user: dict = Depends(require_role("user"))):
    _log_request(request, user)
    service = request.path_params["service_name"].upper()
    upstream = os.environ.get(f"UPSTREAM_{service}_URL")
    if not upstream:
        raise HTTPException(status_code=502, detail="Upstream not configured")
    resp = await _proxy_post(upstream, request, user)
    _log_response(request, resp.status_code)
    return resp


@app.get("/api/v1/route/{service_name}")
async def route_get_v1(request: Request, user: dict = Depends(require_role("user"))):
    _log_request(request, user)
    service = request.path_params["service_name"].upper()
    upstream = os.environ.get(f"UPSTREAM_{service}_URL")
    if not upstream:
        raise HTTPException(status_code=502, detail="Upstream not configured")
    resp = await _proxy_get(upstream, request, user)
    _log_response(request, resp.status_code)
    return resp


# PDF upload endpoint that proxies file to CIM pipeline
@app.post("/api/v1/upload/pdf")
async def upload_pdf(request: Request, file: UploadFile = File(...), user: dict = Depends(require_role("service_account"))):
    _log_request(request, user)
    upstream = os.environ.get("UPSTREAM_CIM_URL")
    if not upstream:
        raise HTTPException(status_code=502, detail="CIM upstream not configured")
    # stream file content to upstream
    files = {"file": (file.filename, await file.read(), file.content_type)}
    token = os.environ.get("GATEWAY_FORWARD_TOKEN")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(f"{upstream.rstrip('/')}/api/v1/upload/pdf", files=files, headers=headers)
        except httpx.RequestError as e:
            LOG.exception("Upstream request failed: %s", e)
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")
    _log_response(request, resp.status_code)
    return JSONResponse(status_code=resp.status_code, content=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text)


# Direct text processing endpoint that proxies to PHI detection service
@app.post("/api/v1/process/text")
async def process_text(request: Request, text: Optional[str] = Form(None), file: Optional[UploadFile] = File(None), user: dict = Depends(require_role("user"))):
    _log_request(request, user)
    upstream = os.environ.get("UPSTREAM_PHI_URL")
    if not upstream:
        raise HTTPException(status_code=502, detail="PHI upstream not configured")
    # prefer direct text
    if text:
        payload = {"text": text}
        token = os.environ.get("GATEWAY_FORWARD_TOKEN")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(f"{upstream.rstrip('/')}/api/v1/detect", json=payload, headers=headers)
            except httpx.RequestError as e:
                LOG.exception("Upstream request failed: %s", e)
                raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")
        _log_response(request, resp.status_code)
        return JSONResponse(status_code=resp.status_code, content=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text)

    # fallback: accept file upload and forward
    if file:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        token = os.environ.get("GATEWAY_FORWARD_TOKEN")
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(f"{upstream.rstrip('/')}/api/v1/detect_file", files=files, headers=headers)
            except httpx.RequestError as e:
                LOG.exception("Upstream request failed: %s", e)
                raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")
        _log_response(request, resp.status_code)
        return JSONResponse(status_code=resp.status_code, content=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text)

    raise HTTPException(status_code=400, detail="No text or file provided")
