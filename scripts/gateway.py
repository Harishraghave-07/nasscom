"""
scripts/gateway.py

Simple gateway CLI to select files for verification and invoke
`scripts/run_verification_attachment.py` logic.

Features:
- Interactive file chooser (lists PDFs in working directory)
- Accepts `--input` to run non-interactively
- Creates an output folder under `debug_output/verification/gateway_run/YYYYMMDD_HHMMSS/`
- Invokes the existing `run_verification_attachment` main() for consistent behavior

Usage:
python3 scripts/gateway.py            # interactive
python3 scripts/gateway.py --input /path/to/file.pdf
"""

import sys
from pathlib import Path
import argparse
import datetime
import shutil
import os
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gateway")

# ensure repo root on path
try:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
except Exception:
    pass


def list_pdfs(cwd: Path) -> list:
    return sorted([p for p in cwd.iterdir() if p.suffix.lower() == ".pdf" and p.is_file()])


def interactive_choose() -> Path:
    cwd = Path.cwd()
    pdfs = list_pdfs(cwd)
    if not pdfs:
        log.info("No PDF files found in %s", cwd)
        path = input("Enter path to PDF file: ").strip()
        return Path(path)

    log.info("Found %d PDF(s) in %s", len(pdfs), cwd)
    for i, p in enumerate(pdfs, start=1):
        print(f"{i}. {p.name}")
    print("0. Enter a custom path")
    sel = input("Choose file number: ").strip()
    try:
        si = int(sel)
        if si == 0:
            path = input("Enter path to PDF file: ").strip()
            return Path(path)
        elif 1 <= si <= len(pdfs):
            return pdfs[si - 1]
    except Exception:
        pass
    # fallback: try to interpret as path
    return Path(sel)


def make_out_dir(base: str = "debug_output/verification/gateway_run") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / ts
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def run_via_runner(input_path: str, out_dir: str) -> int:
    """Invoke the existing runner module by calling its main() after setting sys.argv."""
    try:
        import scripts.run_verification_attachment as runner_mod
    except Exception:
        # last-resort: try importing by module name via importlib
        try:
            from importlib import import_module
            runner_mod = import_module("scripts.run_verification_attachment")
        except Exception as e:
            log.exception("Failed to import runner module: %s", e)
            return 2

    # call main with controlled args
    old_argv = sys.argv.copy()
    sys.argv = ["run_verification_attachment.py", "--input", str(input_path), "--out", str(out_dir)]
    try:
        runner_mod.main()
        return 0
    except SystemExit as se:
        # runner uses sys.exit; capture code
        code = int(se.code) if isinstance(se.code, int) else 0
        return code
    except Exception as e:
        log.exception("Runner failed: %s", e)
        return 1
    finally:
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to a PDF to verify")
    parser.add_argument("--force-blackbox", action="store_true", help="Force blackbox redaction style for debugging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--auto-out", action="store_true", help="Use auto-generated output dir under debug_output")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # allow quick override for redaction style to force visual black boxes
    if args.force_blackbox:
        try:
            from src.core.config import SETTINGS

            SETTINGS.mask.redaction_style = "blackbox"
            log.info("Overrode redaction style to blackbox for debugging")
        except Exception:
            log.exception("Failed to override SETTINGS for blackbox")

    if args.input:
        in_path = Path(args.input)
    else:
        in_path = interactive_choose()

    if not in_path.exists():
        log.error("File not found: %s", in_path)
        return

    if args.auto_out:
        out_dir = make_out_dir()
    else:
        out_dir = make_out_dir()

    log.info("Running verification for %s -> %s", in_path, out_dir)
    code = run_via_runner(str(in_path), out_dir)
    if code == 0:
        log.info("Verification completed; results in %s", out_dir)
    else:
        log.error("Verification failed with exit code %d; see logs", code)


if __name__ == "__main__":
    main()
