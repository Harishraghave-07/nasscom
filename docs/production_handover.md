## Production Handover — PII Masking Service

This document is a concise handover for operations teams responsible for running and maintaining the PII masking pipeline in production.

### Service Overview
- Purpose: The PII Masking Service ingests clinical images and documents, detects potential PII/PHI, and applies masking (black-bar, blur, or inpainting) to remove or obfuscate sensitive information before downstream storage or processing.
- High-level stages: OCR -> PHI detection (legacy detectors ± Presidio) -> Mask synthesis (inpainting/black-bar) -> Audit logging (JSONL) -> Delivery.

### Key Configuration Files
- `config.yaml` (repo root): primary runtime configuration. Critical keys:
  - `presidio_canary_percentage` (0-100): per-image probability Presidio analyzer is used instead of legacy detectors. Configure this in the central `AppConfig` (for example via `SETTINGS.presidio_canary_percentage` or environment variables) rather than ad-hoc `config.yaml` copies.
  - `shadow_mode_enabled` (bool): when true and Presidio is executed, Presidio's output is logged to `debug_output/shadow_audit/` but legacy output remains authoritative.
  - `mask_type`: controls masking style (`black_bar`, `blur`, `inpaint`).
  - `logging.level` and `debug_output` paths.

Open and review `config.yaml` before changing production behavior. Small changes can change audit and detection results.

### Monitoring & Dashboards
- Grafana dashboard (recommended import): `grafana/pii_masking_dashboard.json` (UID: `pii-masking-overview`). Import this JSON into Grafana, or use the dashboard UID if hosted.
- Key metrics to scrape from the service (Prometheus style):
  - `pii_processing_duration_seconds` (histogram)
  - `pii_processed_total` (counter)
  - `pii_errors_total` (counter)
  - `pii_detections_total{pii_type=...}` (counter)

Quick checks (examples):
```bash
# Tail recent audit logs
tail -n 200 debug_output/pdf_audit/$(date +%F).jsonl

# Tail shadow audit (presidio vs legacy)
tail -n 200 debug_output/shadow_audit/$(date +%F).jsonl

# Run the local analyzer script to summarize human feedback
python3 scripts/analyze_feedback.py feedback.csv --out reports/feedback_summary.json
```

### Alerting & On-Call Procedure
Alert rules should be configured in Grafana/Prometheus for:
- High error rate: `increase(pii_errors_total[5m]) / increase(pii_processed_total[5m]) > X` (e.g., X = 0.05)
- High latency: `histogram_quantile(0.95, sum(rate(pii_processing_duration_seconds_bucket[5m])) by (le)) > threshold` (e.g., threshold = 3s)
- Throughput drop: `rate(pii_processed_total[5m])` drops to near-zero unexpectedly.

On-call runbook (summary):
1. Acknowledge alert and open the incident channel (Slack / PagerDuty).
2. Check recent errors and stack traces in logs:
   - App logs (stdout/stderr) and `debug_output/*` JSONL files.
3. If the alert is `High Latency`:
   - Check worker CPU/RAM, disk I/O, and OCR subsystem (EasyOCR) utilization.
   - Consider scaling horizontally (start additional worker instances) or throttling input.
4. If alert is `High Error Rate`:
   - Inspect error messages in `debug_output/pdf_audit/*.jsonl`.
  - If it is an external library (Presidio, OCR) failure, roll back the last deployment and notify engineering.
  - Note: Presidio activation and canary percentage are now centralized in `AppConfig`. Search-and-replace or direct edits to `config.yaml` for presidio flags are deprecated.
5. If false-positive or false-negative spikes are detected:
   - Review `debug_output/shadow_audit/` to compare legacy and presidio outputs.
   - Run `scripts/analyze_feedback.py` on reviewer CSVs to prioritize rule adjustments.

Escalation: If the issue cannot be resolved within 30 minutes, escalate to the on-call engineer for the platform and notify product/engineering leads.

### How to Deploy a New Version
This repository uses GitHub Actions for CI. Typical deploy flow:
1. Create a feature branch and open a pull request targeting `main`.
2. Ensure unit tests pass and static checks (CI) run in the PR. Fix issues if CI fails.
3. Merge the PR to `main`.
4. Tag a release (semantic version tag), e.g.: `git tag -a v1.2.0 -m "release v1.2.0" && git push origin v1.2.0`.
5. The release pipeline (CI) will run unit and integration checks and then trigger the deployment pipeline (CD) that deploys artifacts to the production environment (Docker image push, Kubernetes rollout, etc.)

Rollback procedure:
- If a new release causes regressions, roll back by redeploying the previous stable tag and open an incident. Example:
```bash
# Roll back to previous tag
kubectl set image deployment/pii-masking pii-masking=registry.example.com/pii-masking:v1.1.3
```

### Troubleshooting Guide
- Problem: Latency is high
  - Symptoms: P95 processing latency > threshold; backlog forms.
  - Quick checks:
    - Inspect system metrics (CPU, RAM, disk, network).
    - Check OCR worker logs and queue lengths.
  - Remedies:
    - Scale workers horizontally.
    - Temporarily set `mask_type` to `black_bar` to reduce expensive inpainting.
    - If OCR is the bottleneck, consider lowering image resolution or using a faster OCR engine.

- Problem: Sudden spike in errors
  - Symptoms: `pii_errors_total` increases quickly.
  - Quick checks:
    - Tail `debug_output/pdf_audit/*.jsonl` for exception traces.
    - Check for recent deployment or config changes.
  - Remedies:
    - If caused by a library upgrade, roll back.
    - If caused by input format change, pause ingestion and investigate sample artifacts.

- Problem: Increase in false positives or false negatives
  - Symptoms: Human reviewer feedback shows many FP/FN.
  - Quick checks:
    - Compare `debug_output/shadow_audit/` to see discrepancy between legacy and Presidio.
    - Run `scripts/analyze_feedback.py` on the reviewer CSV to get top offenders.
  - Remedies:
    - Adjust confidence thresholds in `config.yaml`.
    - Add targeted custom rules to `src/integration/policy_matrix.py` or policy config to suppress common false positives.
    - Increase canary percentage gradually if using Presidio, and monitor shadow logs.

- Problem: Disk fills up due to audit logs
  - Symptoms: host disk usage high, writes failing.
  - Quick checks:
    - `du -sh debug_output/*` and `ls -lh debug_output/`
  - Remedies:
    - Delete or archive old JSONL files to object storage.
    - Implement a log rotation/TTL policy for `debug_output/`.

### Useful repository commands
```bash
# Run unit tests (local)
pytest -q

# Lint / static checks
flake8 src tests

# Run analyzer script for reviewer feedback
python3 scripts/analyze_feedback.py feedback.csv --out reports/feedback_summary.json

# Import Grafana dashboard (if hosting Grafana locally)
# (Use Grafana UI -> Dashboards -> Import and upload grafana/pii_masking_dashboard.json)
```

### Contacts
- Primary on-call: platform team (pager duty)
- Secondary: engineering leads (see team on-call roster)

---
Document owner: SRE / Platform
Last updated: 2025-09-10
