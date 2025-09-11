"""Configuration management for the Clinical Image Masker (CIM).

This module centralizes configuration for OCR, PHI detection, masking,
processing, logging, and application-level settings using Pydantic's
BaseSettings so environment variables and a `.env` file can override values.

Clinical & compliance notes
--------------------------
- Logs MUST avoid containing raw PHI. The logging configuration here includes
  a redaction formatter that masks common PHI patterns before writing to disk.
- Audit logging should be enabled in production and persisted to protected
  storage. The `LoggingConfig.enable_audit_trail` flag controls audit logging.
- Temporary files should be stored in a controlled `temp_dir` and cleaned up
  regularly. `cleanup_expired_files` removes stale files older than
  `data_retention_days`.

This file is intentionally conservative and performs validation checks that
may perform network or filesystem calls — call them during startup and not at
import time for best results in test environments.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import shutil
import socket
import tempfile
import urllib.request
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, ValidationError, validator

# pydantic v2 moved BaseSettings to the `pydantic-settings` package. Try to
# import the official BaseSettings; if not available, fall back to using
# pydantic.BaseModel as a shim and load a local .env manually into
# os.environ so configuration values are still picked up from environment.
try:
    from pydantic_settings import BaseSettings  # type: ignore
    _BASESETTINGS_IMPL = "pydantic_settings"
except Exception:
    # fallback shim: use BaseModel and populate os.environ from .env
    from pydantic import BaseModel as BaseSettings  # type: ignore
    _BASESETTINGS_IMPL = "shim"

    # Manual .env loader (minimal). This avoids adding python-dotenv as a
    # hard dependency for CI runs. Load KEY=VALUE pairs and export them if
    # not already present in the environment.
    try:
        env_path = Path(".env")
        if env_path.exists():
            for raw in env_path.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
    except Exception:
        # Non-fatal: if reading .env fails, environment variables may still
        # be provided by the execution environment.
        pass


DEFAULT_PHI_REGEXES = [
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN-like
    r"\b\+?\d{1,3}[ -]?\(?\d{2,3}\)?[ -]?\d{3}[ -]?\d{4}\b",  # phone
    r"[A-Z][a-z]+\s+[A-Z][a-z]+",  # simple name heuristic
]


class OCRConfig(BaseModel):
    """OCR-related configuration.

    confidence_threshold: tokens below this are ignored. Helps reduce false
    positives from OCR in clinical scans.
    """

    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0)
    min_text_area: int = Field(100, ge=0)
    supported_languages: List[str] = Field(default_factory=lambda: ["en"])
    gpu_enabled: bool = False
    batch_size: int = Field(1, ge=1)
    # allow a small slack below the configured confidence threshold so we
    # don't discard marginally-lower OCR regions; this helps avoid empty
    # accepted region lists when OCR confidences are slightly below target.
    confidence_slack: float = Field(0.1, ge=0.0, le=1.0)
    # fuzzy matching threshold (0-100) used to match OCR token sequences to
    # target PHI phrases when attempting to merge adjacent tokens into a
    # single region. Higher means stricter matching. Default 80 for parity
    # with common fuzzywuzzy thresholds.
    fuzz_threshold: int = Field(80, ge=0, le=100)
    # allow the system to operate in a fallback/CI mode where heavy
    # dependencies (EasyOCR / torch) may be unavailable; detectors should
    # check this flag to degrade gracefully.
    enable_fallback: bool = True
    # When True, critical OCR post-processing failures (like merging)
    # will raise and surface to the caller instead of falling back silently.
    fail_on_merge_error: bool = True


class PHIDetectionConfig(BaseModel):
    """PHI detection settings.

    custom_phi_patterns: a mapping from identifier name to regex pattern. These
    patterns are applied in addition to model-based detection.
    """

    enable_regex_patterns: bool = True
    enable_spacy_ner: bool = True
    spacy_model_name: str = "en_core_web_sm"
    custom_phi_patterns: Dict[str, str] = Field(default_factory=dict)
    phi_confidence_threshold: float = Field(0.8, ge=0.0, le=1.0)
    enable_audit_logging: bool = True
    # Whether audit logs should redact detected text snippets. When True,
    # detections in persisted audit records will have their `text` field
    # replaced with a placeholder to reduce risk of PHI leakage in logs.
    audit_redact_snippets: bool = Field(False)
    # When True, also publish a separate non-redacted audit stream for
    # trusted internal review. This should only be used in controlled
    # environments where access to raw PHI in logs is strictly audited.
    audit_allow_nonredacted: bool = Field(False)
    # Name of the non-redacted audit logger when `audit_allow_nonredacted`
    # is enabled. The application logging config can route this logger to
    # a protected sink.
    nonredacted_audit_logger_name: str = Field("cim.audit.phi_raw")
    # Backwards-compatible flag used by some tests and legacy code paths to
    # toggle whether a Presidio-based detection pipeline should be used.
    # Kept here to avoid import-time failures when older code sets this
    # attribute dynamically during testing.
    use_presidio: bool = Field(False)
    # Legacy compatibility flags used in older tests and code paths. Keeping
    # them here avoids ValueError when tests set these attributes dynamically.
    use_spacy_fallback: bool = Field(True)
    use_regex_fallback: bool = Field(True)

    @validator("custom_phi_patterns", pre=True)
    def _ensure_valid_regexes(cls, v):
        if not isinstance(v, dict):
            raise ValueError("custom_phi_patterns must be a dict of name -> regex")
        # validate regex compilation
        for name, pattern in v.items():
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex for {name}: {e}")
        return v


class MaskingConfig(BaseModel):
    """Masking and inpainting configuration.

    inpainting_method: one of telea or ns (Navier-Stokes approximation in OpenCV).
    """

    inpainting_method: str = Field("telea")
    inpainting_radius: int = Field(3, ge=1, le=20)
    mask_expansion_pixels: int = Field(5, ge=0, le=50)
    preserve_aspect_ratio: bool = True
    output_quality: int = Field(95, ge=1, le=100)
    # Temporary directory to store caching/intermediate files for masking
    # operations. Providing a default ensures ImageInpainter can create a
    # cache directory in environments where AppConfig isn't fully wired.
    temp_dir: str = Field("temp/")
    # allow a fallback mode where inpainting may use simplified algorithms
    # or mocked implementations for testing.
    enable_fallback: bool = False
    # redaction style: 'inpaint' uses inpainting algorithms; 'blackbox' draws solid black boxes
    # 'blackbox_merge' draws a single merged black bar covering all detected regions
    redaction_style: str = Field("inpaint")
    # padding in pixels when merging bboxes for a single black bar
    blackbox_padding_pixels: int = Field(5, ge=0, le=200)
    # surgical masking padding (small per-entity padding when using 'surgical' style)
    surgical_padding_pixels: int = Field(2, ge=0, le=50)
    # language-aware padding heuristics
    lang_padding_default: int = Field(2, ge=0, le=100, description="Default surgical padding in pixels")
    lang_padding_cjk: int = Field(6, ge=0, le=200, description="Surgical padding for CJK or dense fonts")
    # allow tuning inpainting radius for dense scripts
    inpainting_radius_cjk: int = Field(5, ge=1, le=50)

    @validator("inpainting_method")
    def _method_allowed(cls, v):
        if v not in ("telea", "ns"):
            raise ValueError("inpainting_method must be 'telea' or 'ns'")
        return v

    @validator("redaction_style")
    def _redaction_allowed(cls, v: str):
        allowed = {"inpaint", "blackbox", "blackbox_merge", "surgical"}
        if v not in allowed:
            raise ValueError(f"redaction_style must be one of {allowed}")
        return v


class ProcessingConfig(BaseModel):
    max_image_size: int = Field(2048, ge=128)
    supported_formats: List[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".dicom"])
    temp_dir: str = Field("temp/")
    max_batch_size: int = Field(10, ge=1)
    timeout_seconds: int = Field(300, ge=1)
    # Allow tests/dev environments to continue when optional heavy
    # dependencies are missing. Set to False in production to fail-fast on
    # missing critical components.
    allow_missing_optional_dependencies: bool = Field(False)

    @validator("temp_dir")
    def _normalize_temp_dir(cls, v: str) -> str:
        return str(Path(v).as_posix())


class LoggingConfig(BaseModel):
    log_level: str = Field("INFO")
    log_file_path: str = Field("logs/clinical_masker.log")
    enable_audit_trail: bool = True
    audit_log_path: str = Field("logs/phi_audit.log")
    max_log_size_mb: int = Field(100, ge=1)
    backup_count: int = Field(5, ge=0)

    @validator("log_level")
    def _valid_level(cls, v):
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR"}
        if v not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v


class AppConfig(BaseSettings):
    """Top-level application configuration.

    Uses BaseSettings so values can be overridden by environment variables or
    a `.env` file. Clinical compliance checks are available via
    `validate_clinical_environment`.
    """

    # environment
    environment: str = Field("dev")
    debug: bool = False

    # security
    secret_key: Optional[str] = None

    # networking
    allowed_hosts: List[str] = Field(default_factory=lambda: ["localhost"])
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:7860"])

    # retention
    data_retention_days: int = Field(30, ge=0)

    # nested configs
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    phi: PHIDetectionConfig = Field(default_factory=PHIDetectionConfig)
    mask: MaskingConfig = Field(default_factory=MaskingConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    # Presidio routing and activation (centralized)
    presidio_canary_percentage: int = Field(
        0, ge=0, le=100, description="Percent of traffic routed to Presidio pipeline"
    )
    use_presidio: bool = Field(False, description="Global toggle to enable Presidio for PHI detection")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("environment")
    def _env_allowed(cls, v):
        allowed = {"dev", "staging", "prod"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

    def get_phi_patterns(self) -> Dict[str, str]:
        """Return a merged dict of PHI detection patterns.

        Combines built-in heuristics with any `custom_phi_patterns` provided in
        the configuration. This is used by detectors and the logging redactor.
        """
        patterns: Dict[str, str] = {f"builtin_{i}": p for i, p in enumerate(DEFAULT_PHI_REGEXES)}
        patterns.update(self.phi.custom_phi_patterns or {})
        return patterns

    @validator("presidio_canary_percentage")
    def _canary_in_range(cls, v):
        if v is None:
            return 0
        if not (0 <= int(v) <= 100):
            raise ValueError("presidio_canary_percentage must be between 0 and 100")
        return int(v)

    @validator("use_presidio")
    def _coherent_use_presidio(cls, v, values):
        # If a canary percentage is set but use_presidio is False, that's likely a config error.
        pct = values.get("presidio_canary_percentage", 0)
        if pct and not v:
            raise ValueError("presidio_canary_percentage > 0 requires use_presidio=True to enable Presidio routing")
        return v

    def validate_clinical_environment(self) -> bool:
        """Run clinical and compliance validations.

        Checks include:
        - Audit logging enabled in production
        - Secret key presence (for prod)
        - Disk space available for expected retention
        - (Optional) network connectivity for model downloads

        Returns True if validations pass, raises ValueError with message on
        failure for clear error propagation during startup.
        """
        # Audit logging requirement
        if self.environment == "prod":
            if not self.logging.enable_audit_trail:
                raise ValueError("Audit trail must be enabled in production (logging.enable_audit_trail)")
            if not self.secret_key:
                raise ValueError("secret_key must be set in production for session management")

        # Disk space check
        temp_path = Path(self.processing.temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(str(temp_path))
        free_gb = usage.free / (1024 ** 3)
        if free_gb < 1:  # require at least 1GB free
            raise ValueError(f"Insufficient disk space in {temp_path}: {free_gb:.2f} GB free")

        # Memory check (best-effort). Only suppress ImportError so other
        # runtime exceptions (for example, failing value checks) surface and
        # cause validation to fail cleanly in tests or production.
        try:
            import psutil
        except ImportError:
            logging.getLogger(__name__).warning("psutil not available; skipping memory check")
        else:
            mem = psutil.virtual_memory()
            free_mem_gb = mem.available / (1024 ** 3)
            if free_mem_gb < 0.5:
                raise ValueError(f"Insufficient available memory: {free_mem_gb:.2f} GB")

        # Network check for model downloads (pypi/spacy)
        try:
            urllib.request.urlopen("https://pypi.org", timeout=3)
        except Exception:
            logging.getLogger(__name__).warning("Network check failed: cannot reach pypi.org; model download validation skipped")

        return True

    def setup_logging(self) -> None:
        """Configure logging for the application with HIPAA-compliant redaction.

        - Creates log directories
        - Adds a RotatingFileHandler for normal logs and (optionally) audit logs
        - Attaches a RedactingFormatter that masks PHI patterns when writing
          to disk.
        """
        # Ensure directories exist
        log_path = Path(self.logging.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path = Path(self.logging.audit_log_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Build redaction regex
        patterns = self.get_phi_patterns()
        combined = "|".join(f"({p})" for p in patterns.values())
        redaction_re = re.compile(combined, flags=re.IGNORECASE)

        class RedactingFormatter(logging.Formatter):
            def format(self, record):
                msg = super().format(record)
                # redact PHI-like patterns
                return redaction_re.sub("[REDACTED_PHI]", msg)

        level = getattr(logging, self.logging.log_level)
        formatter = RedactingFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=self.logging.max_log_size_mb * 1024 * 1024,
            backupCount=self.logging.backup_count,
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)

        root = logging.getLogger()
        root.setLevel(level)
        # Avoid adding duplicate handlers during tests or repeated calls
        if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == handler.baseFilename for h in root.handlers):
            root.addHandler(handler)

        # Setup audit log handler if enabled
        if self.logging.enable_audit_trail:
            audit_handler = RotatingFileHandler(
                filename=str(audit_path),
                maxBytes=self.logging.max_log_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count,
            )
            audit_handler.setLevel(logging.INFO)
            # audit logs should also be redacted
            audit_handler.setFormatter(formatter)
            if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == audit_handler.baseFilename for h in root.handlers):
                root.addHandler(audit_handler)

    def get_temp_file_path(self, prefix: str = "cim_") -> str:
        """Create and return a temporary file path inside the configured temp_dir.

        This ensures temporary files stay under controlled directories that can
        be monitored and cleaned according to policy.
        """
        temp_dir = Path(self.processing.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        fh, path = tempfile.mkstemp(prefix=prefix, dir=str(temp_dir))
        os.close(fh)
        return path

    def cleanup_expired_files(self) -> None:
        """Remove files older than `data_retention_days` in the temp directory.

        This is a safety measure to avoid accidental long-term storage of PHI
        in temporary locations.
        """
        temp_dir = Path(self.processing.temp_dir)
        if not temp_dir.exists():
            return
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=self.data_retention_days)
        for p in temp_dir.iterdir():
            try:
                mtime = datetime.datetime.utcfromtimestamp(p.stat().st_mtime)
                if mtime < cutoff:
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p)
            except Exception:
                logging.getLogger(__name__).exception(f"Failed to clean {p}")


# Provide a single, importable settings instance for application use
try:
    SETTINGS = AppConfig()
except ValidationError as e:
    # When running in CI or during tests the environment may be incomplete.
    # Raise a clear exception so callers can handle startup failures.
    raise


__all__ = ["AppConfig", "SETTINGS", "OCRConfig", "PHIDetectionConfig", "MaskingConfig", "ProcessingConfig", "LoggingConfig"]
