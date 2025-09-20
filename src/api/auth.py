"""Authentication and authorization helpers for the API gateway.

Provides:
- JWT encode/verify utilities
- API key validation for service accounts
- FastAPI dependencies to require roles
- Simple in-memory rate limiter keyed by principal+endpoint
- Audit logging hooks that redact secrets
"""
from __future__ import annotations

import time
import logging
from typing import Dict, Optional, Callable

from fastapi import Depends, HTTPException, Request, status

try:
    from jose import jwt, JWTError
except Exception:
    jwt = None
    JWTError = Exception

from src.core.config import SETTINGS

_auth_logger = logging.getLogger(SETTINGS.auth_audit_logger)
_app_logger = logging.getLogger("cim.auth")

# Simple in-memory rate limiter state: {(key, endpoint): [timestamps]}
_rate_state: Dict[str, list] = {}


def _audit_event(event: str, principal: Optional[str] = None, details: Optional[dict] = None) -> None:
    try:
        payload = {"event": event, "principal": principal or "anonymous", "ts": time.time()}
        if details:
            # redact secrets if configured
            if SETTINGS.auth_log_redact_secrets:
                d2 = {k: ("[REDACTED]" if "secret" in k.lower() or "token" in k.lower() else v) for k, v in details.items()}
            else:
                d2 = details
            payload["details"] = d2
        _auth_logger.info(payload)
    except Exception:
        _app_logger.exception("Failed to write auth audit event")


def create_jwt(payload: Dict, expiry: Optional[int] = None) -> str:
    if jwt is None:
        raise RuntimeError("python-jose (jose) is required for JWT operations")
    secret = SETTINGS.jwt_secret or ""
    if not secret:
        raise RuntimeError("JWT secret not configured")
    exp = int(time.time()) + (expiry or SETTINGS.jwt_exp_seconds)
    to_sign = dict(payload)
    to_sign.update({"exp": exp})
    token = jwt.encode(to_sign, secret, algorithm=SETTINGS.jwt_algorithm)
    _audit_event("token_issued", principal=payload.get("sub"), details={"alg": SETTINGS.jwt_algorithm})
    return token


def verify_jwt(token: str) -> Dict:
    if jwt is None:
        raise RuntimeError("python-jose (jose) is required for JWT operations")
    secret = SETTINGS.jwt_secret or ""
    if not secret:
        raise RuntimeError("JWT secret not configured")
    try:
        decoded = jwt.decode(token, secret, algorithms=[SETTINGS.jwt_algorithm])
        _audit_event("token_verified", principal=decoded.get("sub"), details={"alg": SETTINGS.jwt_algorithm})
        return decoded
    except JWTError as e:
        _audit_event("token_verify_failed", principal=None, details={"error": str(e)})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def validate_api_key(key: str) -> Dict:
    # load configured API keys
    km = SETTINGS.load_api_key_map()
    if not km:
        _audit_event("api_key_missing_config")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="API key store not configured")
    meta = km.get(key)
    if not meta:
        _audit_event("api_key_invalid", details={"key": "[REDACTED]"})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    _audit_event("api_key_valid", principal=meta.get("principal") or meta.get("role"), details={"role": meta.get("role")})
    return meta


def _rate_limit_check(principal: str, endpoint: str, role: str) -> None:
    # Determine RPM by role
    rpm = SETTINGS.rate_limit_user_rpm
    if role == "admin":
        rpm = SETTINGS.rate_limit_admin_rpm
    elif role == "service_account":
        rpm = SETTINGS.rate_limit_service_rpm

    key = f"{principal}:{endpoint}"
    now = time.time()
    window = 60.0
    records = _rate_state.setdefault(key, [])
    # drop old
    while records and records[0] < now - window:
        records.pop(0)
    if len(records) >= rpm:
        _audit_event("rate_limited", principal=principal, details={"endpoint": endpoint, "role": role})
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
    records.append(now)


def require_role(min_role: str) -> Callable:
    """FastAPI dependency generator to require a minimum role.

    Roles hierarchy: admin > service_account > user
    """
    roles = {"user": 1, "service_account": 2, "admin": 3}

    async def _dep(request: Request) -> Dict:
        # If role-based auth is disabled via feature flag, allow anonymous access
        if not getattr(SETTINGS, "enable_role_based_auth", True):
            _audit_event("auth.disabled", principal="system", details={"endpoint": request.url.path})
            return {"principal": "anonymous", "role": "user"}

        # Attempt JWT first (Authorization: Bearer ...)
        auth = request.headers.get("Authorization") or ""
        principal = "anonymous"
        role = "user"
        if auth.startswith("Bearer "):
            token = auth.split(" ", 1)[1].strip()
            try:
                decoded = verify_jwt(token)
                principal = decoded.get("sub") or principal
                role = decoded.get("role") or role
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth token")
        else:
            # try API key header
            key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
            if key:
                meta = validate_api_key(key)
                principal = meta.get("principal") or principal
                role = meta.get("role") or role

        # enforce role hierarchy
        if roles.get(role, 0) < roles.get(min_role, 0):
            _audit_event("access_denied", principal=principal, details={"required": min_role, "actual": role})
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient privileges")

        # rate limit check
        try:
            _rate_limit_check(principal, request.url.path, role)
        except HTTPException:
            raise

        _audit_event("access_granted", principal=principal, details={"role": role, "endpoint": request.url.path})
        return {"principal": principal, "role": role}

    return _dep
