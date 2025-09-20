# API Gateway

This lightweight FastAPI gateway provides:
- Role-based JWT auth (expects `roles` in token payload)
- Simple in-memory rate limiting (configurable via env vars)
- Routing to upstream services via environment-configured URLs

Configuration (environment variables):

- `GATEWAY_JWT_SECRET` - secret used to verify HS256 JWTs (default `changeme`)
- `GATEWAY_JWT_ALGORITHM` - JWT algorithm (default `HS256`)
- `GATEWAY_RATE_LIMIT` - requests allowed per `GATEWAY_RATE_PERIOD` (default `60`)
- `GATEWAY_RATE_PERIOD` - window in seconds for rate limiting (default `60`)
- `UPSTREAM_{SERVICE}_URL` - upstream URL for service routing (e.g. `UPSTREAM_PHI_SERVICE_URL`)

Run locally (example):

```bash
export GATEWAY_JWT_SECRET=supersecret
export UPSTREAM_PHI_SERVICE_URL=http://localhost:8001/detect
uvicorn src.api.gateway:app --host 0.0.0.0 --port 8080
```

Endpoints:

- `GET /health` - gateway health
- `POST /route/{service_name}` - proxy POST to upstream (requires `Authorization: Bearer <token>`)
- `GET /route/{service_name}` - proxy GET to upstream (requires `Authorization: Bearer <token>`)

Notes:
- This gateway is intentionally minimal. For production use, replace the in-memory rate limiter with a distributed store (Redis) and add TLS, logging, monitoring, and proper error handling.
