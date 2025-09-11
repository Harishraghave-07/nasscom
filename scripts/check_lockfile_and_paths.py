#!/usr/bin/env python3
"""
Simple check script for CI to enforce deterministic lockfiles and
prevent committed absolute user paths.

Exits with non-zero if checks fail.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = ROOT / "requirements.lock"

errors = []

if not LOCKFILE.exists():
    errors.append(f"Missing lockfile: {LOCKFILE.name}. Generate it with pip-compile and commit it.")

# Scan for obvious absolute user paths in repository files (naive grep)
# Look for common user-home path prefixes. This is intentionally conservative.
bad_patterns = ["/Users/", "/home/", "C:\\Users\\"]
for p in ROOT.rglob("**/*"):
    try:
        if p.is_file():
            txt = p.read_text(errors="ignore")
            for pat in bad_patterns:
                if pat in txt:
                    errors.append(f"Found absolute path pattern '{pat}' in {p}")
    except Exception:
        pass

if errors:
    print("Lockfile & path check failed:\n")
    for e in errors:
        print(" -", e)
    sys.exit(2)

print("Lockfile & path check passed.")
sys.exit(0)
