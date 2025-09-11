"""Configuration validator and fixer for Clinical Image Masker.

Provides ConfigValidator which validates environment, dependencies,
configuration files, file permissions, model availability, and sample
processing. Offers automatic fixes for common issues and can generate a
minimal working configuration.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

# Heavy dependencies (EasyOCR, spaCy) are imported lazily inside methods that
# use them. This keeps module import fast and reliable in test/CI where these
# packages may not be installed.

LOGGER = logging.getLogger("ConfigValidator")
logging.basicConfig(level=logging.INFO)


class ConfigValidator:
    def __init__(self):
        self.project_root = Path.cwd()
        self.reports_dir = self.project_root / "debug_output" / "config_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def validate_complete_setup(self, config_path: str | None = None) -> Dict[str, Any]:
        report = {
            "environment": self.validate_environment(),
            "dependencies": self.validate_dependencies(),
            "configuration": self.validate_configuration(config_path),
            "file_permissions": self.validate_file_permissions(),
            "model_availability": self.validate_models(),
            "sample_processing": self.validate_sample_processing(),
        }
        out = self.reports_dir / "complete_validation.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        return report

    def validate_environment(self) -> Dict[str, Any]:
        out = {"python_version": sys.version, "cwd": str(self.project_root)}
        # check common env vars
        out["env"] = {k: os.environ.get(k) for k in ["PYTHONPATH", "VIRTUAL_ENV"]}
        return out

    def validate_dependencies(self) -> Dict[str, Any]:
        deps: Dict[str, Any] = {}
        # EasyOCR (lazy import and best-effort test)
        try:
            import easyocr as _easyocr  # type: ignore
            reader = _easyocr.Reader(["en"], gpu=False, verbose=False)
            test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
            if cv2 is not None:
                cv2.putText(test_image, "TEST", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            results = reader.readtext(test_image)
            detected = [r[1] for r in results] if results else []
            deps["easyocr"] = {"status": "OK", "can_detect_text": len(detected) > 0, "detected_text": detected}
        except Exception as e:
            deps["easyocr"] = {"status": "FAILED", "error": str(e)}

        # OpenCV
        try:
            if cv2 is None:
                raise ImportError("cv2 not available")
            test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            mask = np.zeros((100, 100), dtype=np.uint8)
            cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)
            inpainted = cv2.inpaint(test_img, mask, 3, cv2.INPAINT_TELEA)
            deps["opencv"] = {"status": "OK", "inpainting_works": inpainted is not None, "version": cv2.__version__}
        except Exception as e:
            deps["opencv"] = {"status": "FAILED", "error": str(e)}

        # spaCy (lazy import)
        try:
            import spacy as _spacy  # type: ignore
            nlp = _spacy.load("en_core_web_sm")
            doc = nlp("John Doe was born on 01/15/1980")
            ents = [(ent.text, ent.label_) for ent in doc.ents]
            deps["spacy"] = {"status": "OK", "model_loaded": True, "sample_entities": ents}
        except Exception as e:
            deps["spacy"] = {"status": "FAILED", "error": str(e)}

        # numpy
        try:
            deps["numpy"] = {"status": "OK", "version": np.__version__}
        except Exception as e:
            deps["numpy"] = {"status": "FAILED", "error": str(e)}

        # write report
        p = self.reports_dir / "dependencies.json"
        p.write_text(json.dumps(deps, indent=2, default=str))
        return deps

    def validate_configuration(self, config_path: str | None = None) -> Dict[str, Any]:
        # attempt to load config module if provided; otherwise check common settings
        result: Dict[str, Any] = {}
        paths_checked = []
        try:
            candidates = [config_path] if config_path else ["config/default.py", "config.py", "src/core/config.py"]
            for c in candidates:
                if c and Path(c).exists():
                    paths_checked.append(c)
            result["paths_checked"] = paths_checked
            # basic sanity: ensure temp/log dirs exist in config
            result["has_temp_dir"] = Path("temp").exists()
            result["has_logs_dir"] = Path("logs").exists()
        except Exception as e:
            result["error"] = str(e)
        p = self.reports_dir / "configuration.json"
        p.write_text(json.dumps(result, indent=2, default=str))
        return result

    def validate_file_permissions(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for d in ["logs", "temp", "data"]:
            p = Path(d)
            if not p.exists():
                res[d] = {"exists": False}
                continue
            st = p.stat()
            res[d] = {"exists": True, "mode": oct(st.st_mode & 0o777), "writable": os.access(p, os.W_OK)}
        p = self.reports_dir / "file_permissions.json"
        p.write_text(json.dumps(res, indent=2, default=str))
        return res

    def validate_models(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        # spaCy model
        try:
            import spacy as _sp  # type: ignore
            try:
                _sp.load("en_core_web_sm")
                res["spacy_en_core_web_sm"] = {"installed": True}
            except Exception as e:
                res["spacy_en_core_web_sm"] = {"installed": False, "error": str(e)}
        except Exception as e:
            res["spacy"] = {"installed": False, "error": str(e)}

        p = self.reports_dir / "models.json"
        p.write_text(json.dumps(res, indent=2, default=str))
        return res

    def validate_sample_processing(self) -> Dict[str, Any]:
        # run a tiny pipeline: load image, detect text (if possible), generate mask
        res: Dict[str, Any] = {}
        try:
            # create tiny synthetic image
            img = np.ones((200, 300, 3), dtype=np.uint8) * 255
            if cv2 is not None:
                cv2.putText(img, "John Doe", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            else:
                res["processing"] = {"error": "cv2 not available"}
                return res

            detections = []
            try:
                import easyocr as _easyocr  # type: ignore
                try:
                    rdr = _easyocr.Reader(["en"], gpu=False, verbose=False)
                    detections = rdr.readtext(img)
                except Exception as e:
                    res["ocr_error"] = str(e)
            except Exception:
                # easyocr not available; report zero detections
                detections = []

            res["detections_count"] = len(detections)
            p = self.reports_dir / "sample_processing.json"
            p.write_text(json.dumps(res, indent=2, default=str))
            return res
        except Exception as e:
            return {"error": str(e)}

    def fix_common_issues(self) -> Dict[str, Any]:
        fixes_applied = []
        required_dirs = [
            "data/raw",
            "data/processed",
            "logs",
            "temp",
            "debug_output",
            "debug_output/stage1",
            "debug_output/stage2",
            "debug_output/stage3",
            "debug_output/stage4",
        ]
        for d in required_dirs:
            p = Path(d)
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
                fixes_applied.append(f"Created directory: {d}")

        # Attempt to download spaCy model if missing
        try:
            try:
                import spacy as _sp  # type: ignore
                try:
                    _sp.load("en_core_web_sm")
                except Exception:
                    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=False)
                    fixes_applied.append("Attempted to download spaCy en_core_web_sm")
            except Exception:
                # spaCy not installed; nothing to do
                pass
        except Exception:
            pass

        # create default conservative config file
        cfg_path = Path("config")
        cfg_path.mkdir(parents=True, exist_ok=True)
        default_cfg = self.generate_minimal_working_config()
        (cfg_path / "default_fixed.py").write_text(default_cfg)
        fixes_applied.append("Wrote config/default_fixed.py")

        out = {"fixes_applied": fixes_applied}
        p = self.reports_dir / "fixes_applied.json"
        p.write_text(json.dumps(out, indent=2, default=str))
        return out

    def generate_minimal_working_config(self) -> str:
        config = """
# Minimal working configuration for Clinical Image Masker

class Config:
    # OCR Settings - Very permissive for initial testing
    OCR_CONFIDENCE_THRESHOLD = 0.01  # Accept almost everything
    OCR_MIN_TEXT_AREA = 25  # Very small minimum
    OCR_LANGUAGES = ['en']
    OCR_GPU_ENABLED = False  # Force CPU

    # PHI Detection - Basic patterns only
    ENABLE_REGEX_PATTERNS = True
    ENABLE_SPACY_NER = True
    SPACY_MODEL = 'en_core_web_sm'

    # Basic PHI patterns
    PHI_PATTERNS = {
        'name': r'[A-Z][a-z]+ [A-Z][a-z]+',
        'mrn': r'MRN[-\s]?\d{6,}',
        'ssn': r'\d{3}[-\s]?\d{2}[-\s]?\d{4}',
        'phone': r'\d{3}[-\s]?\d{3}[-\s]?\d{4}'
    }

    # Masking Settings - Simple and reliable
    MASKING_METHOD = 'rectangle'
    INPAINTING_METHOD = 'telea'
    MASK_EXPANSION_PIXELS = 3

    # Logging
    LOG_LEVEL = 'DEBUG'
    ENABLE_AUDIT_LOGGING = True

    # File paths
    TEMP_DIR = 'temp'
    LOG_DIR = 'logs'
    DEBUG_OUTPUT_DIR = 'debug_output'
"""
        return config


if __name__ == "__main__":
    cv = ConfigValidator()
    print("Running full validation...")
    rep = cv.validate_complete_setup()
    print(json.dumps(rep, indent=2)[:1000])
