
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
