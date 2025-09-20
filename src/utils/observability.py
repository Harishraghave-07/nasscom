"""Observability helpers: structured logging, correlation IDs, Prometheus metrics,
and safe audit logging without PHI.

This module is dependency-tolerant: it will fall back gracefully if optional
libraries (prometheus_client, pythonjsonlogger) are not installed.
"""
from __future__ import annotations

import time
import uuid
import logging
import json
import functools
from typing import Callable, Any, Dict

try:
    from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry
    from prometheus_client import CONTENT_TYPE_LATEST
    PROM_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    PROM_AVAILABLE = False

try:
    from pythonjsonlogger import jsonlogger
    JSONLOGGER = True
except Exception:
    JSONLOGGER = False

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# HTTP metrics
if PROM_AVAILABLE:
    REGISTRY = CollectorRegistry()
    HTTP_REQUEST_COUNT = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "http_status", "role"],
        registry=REGISTRY,
    )
    HTTP_REQUEST_LATENCY = Histogram(
        "http_request_latency_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint", "role"],
        registry=REGISTRY,
    )
    INFERENCE_LATENCY = Histogram(
        "phi_inference_latency_seconds",
        "PHI model inference latency (seconds)",
        ["model"],
        registry=REGISTRY,
    )
    INFERENCE_ERRORS = Counter(
        "phi_inference_errors_total",
        "PHI inference errors",
        ["model", "error_type"],
        registry=REGISTRY,
    )
else:
    REGISTRY = None
    HTTP_REQUEST_COUNT = HTTP_REQUEST_LATENCY = INFERENCE_LATENCY = INFERENCE_ERRORS = None

# Logger setup
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        if JSONLOGGER:
            fmt = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s %(correlation_id)s")
            handler.setFormatter(fmt)
        else:
            # simple JSON-like formatter
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# Audit logger (writes non-PHI structured audit records to a file)
def get_audit_logger(path: str = "phi_audit.log") -> logging.Logger:
    logger = logging.getLogger("phi_audit")
    if not logger.handlers:
        fh = logging.FileHandler(path)
        if JSONLOGGER:
            fmt = jsonlogger.JsonFormatter()
            fh.setFormatter(fmt)
        else:
            fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    return logger

AUDIT_LOGGER = get_audit_logger()

# Correlation ID middleware
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # try common headers first
        rid = request.headers.get("x-correlation-id") or request.headers.get("x-request-id")
        if not rid:
            rid = str(uuid.uuid4())
        request.state.correlation_id = rid
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = rid
        return response

# Prometheus middleware
class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        endpoint = request.url.path
        method = request.method
        role = getattr(request.state, "principal_role", "anonymous")
        try:
            resp = await call_next(request)
            status = str(resp.status_code)
        except Exception:
            status = "500"
            raise
        finally:
            elapsed = time.time() - start
            if PROM_AVAILABLE:
                try:
                    HTTP_REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status, role=role).inc()
                    HTTP_REQUEST_LATENCY.labels(method=method, endpoint=endpoint, role=role).observe(elapsed)
                except Exception:
                    # metrics must never break the request
                    pass
        return resp

# Expose metrics text for /metrics endpoint
def metrics_response():
    if PROM_AVAILABLE and REGISTRY is not None:
        return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
    return b"", "text/plain"

# Audit event writer - redacts fields that likely contain PHI
_REDACT_KEYS = {"text", "raw", "image", "phi", "ocr_text", "page_image"}

def _redact(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (obj or {}).items():
        if k.lower() in _REDACT_KEYS:
            out[k] = "<REDACTED>"
        else:
            try:
                json.dumps(v)
                out[k] = v
            except Exception:
                out[k] = str(v)
    return out

def audit_event(action: str, principal: Dict[str, Any], details: Dict[str, Any] | None = None):
    # principal should be a dict with non-sensitive identity fields (id, role)
    record = {
        "ts": int(time.time()),
        "action": action,
        "correlation_id": principal.get("correlation_id") if isinstance(principal, dict) else None,
        "principal": {k: principal.get(k) for k in ("id", "name", "role") if isinstance(principal, dict) and k in principal},
        "details": _redact(details or {}),
    }
    try:
        AUDIT_LOGGER.info(json.dumps(record))
    except Exception:
        # best-effort
        AUDIT_LOGGER.info(str(record))

# Decorator to measure inference latency and record errors
def measure_inference(model_name: str):
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                res = fn(*args, **kwargs)
                return res
            except Exception as e:
                if PROM_AVAILABLE:
                    try:
                        INFERENCE_ERRORS.labels(model=model_name, error_type=type(e).__name__).inc()
                    except Exception:
                        pass
                raise
            finally:
                elapsed = time.time() - start
                if PROM_AVAILABLE:
                    try:
                        INFERENCE_LATENCY.labels(model=model_name).observe(elapsed)
                    except Exception:
                        pass
        return wrapper
    return decorator
