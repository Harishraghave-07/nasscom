FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps required by some wheels (opencv, torch may need libstdc++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libsm6 \
    libxrender1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy minimal project files
COPY requirements-core.txt /app/
COPY requirements-presidio.txt /app/
COPY . /app/

# Create a venv and install core deps first
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip wheel
RUN /opt/venv/bin/pip install -r /app/requirements-core.txt

# Optional heavy deps - install on demand to avoid bloating base image
RUN /opt/venv/bin/pip install -r /app/requirements-presidio.txt || echo "Optional heavy deps may fail in constrained environments"

ENV PATH="/opt/venv/bin:$PATH"

# Default command: run tests (CI can override)
CMD ["pytest", "-q"]

# syntax=docker/dockerfile:1

# =============================
# STAGE 1: Build Dependencies
# =============================
FROM python:3.11-slim-bullseye AS builder

# Install system dependencies required for healthcare/ML apps
RUN apt-get update && apt-get install -y --no-install-recommends \
	libgl1-mesa-glx \
	libglib2.0-0 \
	libsm6 \
	libxext6 \
	libxrender-dev \
	libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements.txt for dependency installation
COPY requirements.txt /app/

# Upgrade pip for latest features and security
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies with reproducibility
RUN pip install --no-cache-dir --force-reinstall -r requirements.txt

# =============================
# STAGE 2: Runtime
# =============================
FROM python:3.11-slim-bullseye AS runtime

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
	curl \
	&& rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN useradd -m -u 1000 clinicaluser
USER clinicaluser

# Set working directory
WORKDIR /app

# Copy source code, excluding tests, docs, notebooks (handled by .dockerignore)
COPY . /app

# Set environment variables for best practices
ENV PYTHONUNBUFFERED=1 \
		PYTHONDONTWRITEBYTECODE=1 \
		LOG_LEVEL=INFO

# Expose Gradio port
EXPOSE 7860

# Health check for Gradio endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
	CMD curl --fail http://localhost:7860/health || exit 1

# Entrypoint for Gradio app
ENTRYPOINT ["python", "-m", "src.ui.gradio_app.main"]
