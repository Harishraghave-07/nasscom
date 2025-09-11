"""Simple CI validation for generated synthetic data.

This script checks that the `synthetic_data` directory exists and contains
at least one PNG file. It exits with non-zero code otherwise so CI fails.
"""
import sys
from pathlib import Path


def main():
    d = Path("synthetic_data")
    if not d.exists() or not d.is_dir():
        print("ERROR: synthetic_data directory not found", file=sys.stderr)
        return 2
    pngs = list(d.glob("*.png"))
    if not pngs:
        print("ERROR: no PNG files found in synthetic_data", file=sys.stderr)
        return 3
    print(f"Found {len(pngs)} synthetic PNG(s):")
    for p in pngs[:10]:
        print(" -", p.name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
