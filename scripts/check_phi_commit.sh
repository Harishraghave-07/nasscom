#!/bin/bash
# Simple pre-commit hook to detect obvious PHI patterns in staged files
set -euo pipefail

STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)
PATTERN='\b\d{3}-\d{2}-\d{4}\b' # SSN pattern
EXIT_CODE=0
for f in $STAGED_FILES; do
  if [ -f "$f" ]; then
    if grep -E -n "$PATTERN" "$f" >/dev/null 2>&1; then
      echo "Potential SSN found in $f. Remove PHI from test fixtures before committing."
      EXIT_CODE=1
    fi
  fi
done
exit $EXIT_CODE
