Gateway usage

Environment variables

- `GATEWAY_JWT_SECRET` : JWT secret used to validate incoming tokens (fallbacks to AppConfig.secret_key)
- `GATEWAY_JWT_ALGORITHM` : JWT algorithm (default HS256)
- `GATEWAY_FORWARD_TOKEN` : Optional token the gateway will add when forwarding requests to upstream services
- `UPSTREAM_CIM_URL` : Base URL of the Clinical Image Masker pipeline (e.g., http://localhost:8001)
- `UPSTREAM_PHI_URL` : Base URL of the PHI detection microservice (e.g., http://localhost:8002)
- `UPSTREAM_{SERVICE}_URL` : Generic per-service upstream URL used by `/api/v1/route/{service_name}`

Example curl: text detection

```bash
curl -X POST \
  -H "Authorization: Bearer <JWT>" \
  -F "text=Patient John Doe DOB 01/01/1980" \
  http://localhost:8000/api/v1/process/text
```

Example curl: upload PDF

```bash
curl -X POST \
  -H "Authorization: Bearer <JWT>" \
  -F "file=@/path/to/document.pdf;type=application/pdf" \
  http://localhost:8000/api/v1/upload/pdf
```

Health checks

- `GET /health` - basic health
- `GET /api/v1/health` - v1 health
- `GET /api/v2/health` - v2 health

Notes

- The gateway uses `SETTINGS.cors_origins` from `src.core.config.SETTINGS` to configure CORS.
- Rate limits are per-role and can be adjusted via `GATEWAY_RATE_ADMIN`, `GATEWAY_RATE_SERVICE`, and `GATEWAY_RATE_USER` environment variables.
- Audit logs are written to the application logger configured by `AppConfig.setup_logging()`; the gateway emits structured JSON objects for request/response events.
