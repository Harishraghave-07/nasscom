"""Run pipeline across a small grid of sensitivity parameters and save results.

For each combination we vary:
- OCR confidence threshold (OCRConfig.confidence_threshold)
- OCR fuzz threshold (OCRConfig.fuzz_threshold)
- PHI detection threshold (PHIDetectionConfig.phi_confidence_threshold)
- Overlap merge threshold used by TextDetector (if available)

Outputs are written to `debug_output/sensitivity_runs/{run_name}/` with a report.json.
"""
from pathlib import Path
import json
import shutil
from datetime import datetime

from src.core.config import AppConfig, OCRConfig, PHIDetectionConfig
from src.core.pipeline import ClinicalImageMaskingPipeline

ROOT = Path(".").resolve()
SYN = ROOT / "synthetic_data"
OUT_ROOT = ROOT / "debug_output" / "sensitivity_runs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# parameter grid (small)
ocr_thresholds = [0.6, 0.45, 0.35]
fuzz_values = [80, 70]
phi_thresholds = [0.8, 0.6]
overlap_thresh = [0.5, 0.3]

imgs = list(SYN.glob("*.png"))
if not imgs:
    print("No synthetic images found in synthetic_data/. Run scripts/generate_fake_phi_data.py first.")
    raise SystemExit(1)

runs = []
for ocr_t in ocr_thresholds:
    for fuzz in fuzz_values:
        for phi_t in phi_thresholds:
            for ot in overlap_thresh:
                run_name = f"ocr{int(ocr_t*100)}_f{fuzz}_phi{int(phi_t*100)}_ov{int(ot*100)}"
                out_dir = OUT_ROOT / run_name
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)

                # configure app
                cfg = AppConfig()
                # override nested configs
                cfg.ocr.confidence_threshold = ocr_t
                cfg.ocr.fuzz_threshold = fuzz
                cfg.phi.phi_confidence_threshold = phi_t

                # inject the overlap threshold into TextDetector via a convenience attr
                # The TextDetector uses an argument overlap_thresh when calling merge_overlapping_regions
                # We'll set a global environment-like attribute the detector can read; it's acceptable for testing.
                # If not used, the detector will fallback to its defaults.
                cfg._test_overlap_thresh = ot

                pipeline = ClinicalImageMaskingPipeline(cfg)

                reports = []
                for img in imgs:
                    stem = img.name
                    out_mask = out_dir / (stem.replace('.png', '_masked.png'))
                    rep = pipeline.process_single_image(str(img), output_path=str(out_mask), metadata={"test_run": run_name})
                    rep["input"] = str(img)
                    rep["masked_output"] = str(out_mask)
                    reports.append(rep)

                combined = {
                    "run": run_name,
                    "params": {"ocr_threshold": ocr_t, "fuzz": fuzz, "phi_threshold": phi_t, "overlap_thresh": ot},
                    "reports": reports,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                (out_dir / "report.json").write_text(json.dumps(combined, indent=2, default=str))
                print("Completed", run_name)
                runs.append(run_name)

print("All runs completed. Results in", str(OUT_ROOT))
