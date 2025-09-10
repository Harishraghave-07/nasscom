# Model Governance — PII Detection

This document describes the model governance controls for the PII detection system. It is intended for engineering, SRE, data science, and auditors.

## Model Versioning

- Version scheme: `pii-detector-v[major].[minor].[patch]`.
  - Major: breaking changes to model contract or labels (requires full re-validation).
  - Minor: new features, retrained models that add capability without breaking backward compatibility.
  - Patch: small fixes, calibration changes, or hyperparameter tuning that do not change labels.
- Each model artifact must be stored in the model registry or artifact storage under its full tag and accompanied by:
  - Training dataset references (commit hash / dataset snapshot ID)
  - Training code/version (git commit SHA)
  - Baseline and validation metrics (precision, recall, F1 by PII type)
  - Responsible owner and change rationale

## Retraining Policy

### Scheduled Retraining
- Routine retraining cadence: every two weeks (14 days). The scheduled retrain should:
  1. Pull the latest curated correction dataset from the human-in-the-loop store.
  2. Run automated hyperparameter tuning (optional) and fine-tuning as described in `scripts/retrain_model.py`.
  3. Evaluate on the canonical validation set and produce a comparison report versus the active production model.
  4. Only models that pass automated acceptance tests (no regressions > configurable thresholds) are forwarded to the staging deployment pipeline.

### Trigger Conditions (Emergency Retrain)
- An emergency retrain may be initiated if any of the following occur:
  - Critical drop in recall for any high-risk PII type (example: PAN, Aadhaar) exceeding 5% absolute drop compared to baseline.
  - Sustained increase in false negatives or a human feedback rate (false negatives) above configured threshold over 24 hours.
  - Security or data-privacy incident that requires retraining to remove a harmful detection pattern.

### Emergency Retrain Procedure
1. On detection of a trigger, the on-call engineer opens an incident and notifies the Data Science lead.
2. Document the issue, collect representative failing samples, and curate a targeted correction dataset.
3. Run the fine-tuning workflow on the targeted dataset and validate strictly against the full validation suite.
4. If validation shows improvement for the affected PII type and no unacceptable regressions elsewhere, the model may proceed to expedited approval (see Deployment Approval Process).

### Acceptance Criteria for New Models
- New candidate models must meet or exceed the following before promotion:
  - F1 improvement on targeted PII type(s) or no degradation greater than configurable tolerance (default: 0.5% absolute) on other PII types.
  - No increase in critical error rate (exceptions) beyond predefined limits.
  - Passing integration tests (inference correctness, latency, memory footprint thresholds).

## Deployment Approval Process

1. Prepare a release package that includes: model artifact (tagged), training metadata, validation metrics, regression report, and deployment plan.
2. Required approvers before production rollout:
   - Lead Data Scientist (validates metrics and model quality)
   - Head of Compliance or delegated compliance officer (validates privacy/regulatory implications)
   - SRE/Platform lead (validates deployment and operational concerns)
3. Approval workflow:
   - Submit the release package to the model registry and open a release request ticket (Jira/GitHub issue) listing approvers.
   - Approvers review artifacts, run any requested local verification steps, and record approval in the ticket.
   - Once all approvals are present, schedule a canary rollout (recommended) and enable `presidio_canary_percentage` per config for gradual traffic migration.
4. Post-deployment checks:
   - Monitor shadow audit logs and real-time metrics for at least 24–72 hours.
   - If regressions are observed, trigger rollback and open an incident.

### Rollback & Audit
- All deployments must include an automated rollback plan and are logged with audit entries including who approved, when, and the validation artifacts.
- Retain model artifacts and validation reports for audit retention period (e.g., 1 year) in model registry or object storage.

---
Document owner: Data Science
Last updated: 2025-09-10
