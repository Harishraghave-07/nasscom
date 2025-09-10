"""
scripts/verify_presidio.py

Quick verification script to run in your development environment.
What it does:
- checks for installed packages: presidio-analyzer, presidio-anonymizer, spacy
- prints versions for those packages
- tries to load a spaCy model (en_core_web_sm)
- instantiates Presidio AnalyzerEngine and AnonymizerEngine (if available)
- runs a sample analyze + anonymize on a short text and prints a compact JSON report

Exit codes:
- 0: all checks passed and sample anonymization succeeded
- 1: non-fatal issues (missing optional components); script still prints suggestions
- 2: fatal error (unable to run sample anonymization when Presidio is expected)

Usage:
python3 scripts/verify_presidio.py

"""

import json
import sys
import traceback

SAMPLE_TEXT = "Patient John Doe, born 01/01/1980, phone +1 555 123 4567, MRN 123456"

report = {
    "env": {},
    "spacy_model": None,
    "presidio_analyzer": None,
    "presidio_anonymizer": None,
    "sample_analyze": None,
    "sample_anonymize": None,
}

exit_code = 0

# Helper to safe import and get version
def safe_import(module_name):
    try:
        mod = __import__(module_name)
        ver = getattr(mod, "__version__", None)
        return mod, ver
    except Exception:
        return None, None

# 1) check basic packages
for pkg in ("presidio_analyzer", "presidio_anonymizer", "spacy"):
    mod, ver = safe_import(pkg)
    report["env"][pkg] = {
        "present": mod is not None,
        "version": ver,
    }

# Try to get importlib.metadata versions for better visibility (py3.8+)
try:
    from importlib.metadata import version, PackageNotFoundError
    for pkg_name in ("presidio-analyzer", "presidio-anonymizer", "spacy"):
        try:
            report["env"][pkg_name] = report["env"].get(pkg_name) or {}
            report["env"][pkg_name]["dist_version"] = version(pkg_name)
        except PackageNotFoundError:
            pass
except Exception:
    # ignore if importlib.metadata isn't usable
    pass

# 2) spaCy model check
try:
    import spacy
    try:
        # try common small model
        model_name = "en_core_web_sm"
        nlp = spacy.load(model_name)
        report["spacy_model"] = {"loaded": True, "model": model_name, "pipeline": nlp.pipe_names}
    except Exception as e:
        report["spacy_model"] = {"loaded": False, "error": str(e)}
        exit_code = max(exit_code, 1)
except Exception as e:
    report["spacy_model"] = {"loaded": False, "error": "spaCy not importable: %s" % str(e)}
    exit_code = max(exit_code, 1)

# 3) Presidio Analyzer + anonymizer smoke test
try:
    from presidio_analyzer import AnalyzerEngine
    analyzer = AnalyzerEngine()
    # run analysis on sample text
    results = analyzer.analyze(text=SAMPLE_TEXT, language="en")
    # Simplify results for report
    report["presidio_analyzer"] = {
        "available": True,
        "num_detections": len(results),
        "detections": [{
            "entity_type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": getattr(r, "score", None),
            "text": SAMPLE_TEXT[r.start:r.end] if (r.start is not None and r.end is not None) else None,
        } for r in results],
    }
except Exception as e:
    report["presidio_analyzer"] = {"available": False, "error": str(e)}
    exit_code = max(exit_code, 1)

try:
    from presidio_anonymizer import AnonymizerEngine
    anonymizer = AnonymizerEngine()
    # If analyzer produced results, try anonymize
    if report.get("presidio_analyzer", {}).get("num_detections", 0) > 0:
        try:
            anonymized = anonymizer.anonymize(text=SAMPLE_TEXT, analyzer_results=results)
            report["presidio_anonymizer"] = {"available": True, "anonymized_text": anonymized.text}
            report["sample_anonymize"] = True
        except Exception as e:
            report["presidio_anonymizer"] = {"available": True, "error": str(e)}
            report["sample_anonymize"] = False
            exit_code = max(exit_code, 2)
    else:
        report["presidio_anonymizer"] = {"available": True, "note": "no detections to anonymize"}
        report["sample_anonymize"] = False
        exit_code = max(exit_code, 1)
except Exception as e:
    report["presidio_anonymizer"] = {"available": False, "error": str(e)}
    exit_code = max(exit_code, 1)

# Print a readable summary
print("\n=== Presidio verification report ===\n")
print(json.dumps(report, indent=2))
print("\nSummary:")
if report["presidio_analyzer"].get("available"):
    print("- presidio-analyzer: OK, detections=%d" % report["presidio_analyzer"].get("num_detections", 0))
else:
    print("- presidio-analyzer: MISSING or failed to initialize")

if report["presidio_anonymizer"].get("available"):
    if report["presidio_anonymizer"].get("anonymized_text"):
        print("- presidio-anonymizer: OK, sample anonymized")
    else:
        print("- presidio-anonymizer: available but sample anonymization not performed")
else:
    print("- presidio-anonymizer: MISSING or failed to initialize")

if report["spacy_model"] and report["spacy_model"].get("loaded"):
    print("- spaCy model: %s loaded (pipeline: %s)" % (report["spacy_model"]["model"], report["spacy_model"]["pipeline"]))
else:
    print("- spaCy model: not loaded or missing")

# Exit with code to allow CI to fail if critical
sys.exit(exit_code)
