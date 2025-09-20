# Deployment & Secret Management Guide

This guide explains how to manage secrets (JWT keys, DB credentials, API keys) for local development and production.

Local (Docker Compose)
- Create a `.env` file at the repo root with the following entries:

```
JWT_SECRET=super-secret-change-me
DB_URL=postgresql://user:password@db:5432/nasscom
SERVICE_API_KEYS=svc1:supersecret1,svc2:supersecret2
```

- Then run:

```bash
docker-compose build --pull
docker-compose up
```

Kubernetes (recommended for production)
- Use K8s secrets rather than embedding secrets into ConfigMaps.

Example to create secret from literals:

```bash
kubectl create secret generic nasscom-secrets \
  --from-literal=JWT_SECRET="$(openssl rand -base64 32)" \
  --from-literal=DB_URL="postgresql://user:password@db:5432/nasscom" \
  -n nasscom
```

- To reference these secrets in the manifests, update `deploy/k8s/*-deployment.yaml` to mount or envFrom the secret. The `deploy/k8s/secret-template.yaml` is a starting point.

External secret stores
- For production, use Vault/ExternalSecrets or cloud KMS to inject secrets into Pods. Do not store secrets in Git.

JWT rotation
- Store current and previous secrets to support token rotation. Use K8s secrets with labels to track versions and update deployments with rolling restarts.
