OCR merge helper and fail-fast behavior

What changed

- Restored TextDetector.merge_overlapping_regions as a bound method on the TextDetector class. This fixes an AttributeError where the pipeline could silently skip merging OCR regions.
- Added a configuration flag: `OCRConfig.fail_on_merge_error` (default: true). When true, errors raised during region merging are re-raised so the pipeline fails fast instead of falling back to unmerged regions.

Why

Merging adjacent/overlapping OCR boxes is critical to accurate PHI localization. Previously a mis-indented duplicate implementation meant the method wasn't bound to the class, which caused merging to silently fail and produced zero-mask coverage for some PDFs. Making merge errors fail-fast helps operators detect and fix OCR/postprocessing problems rather than silently producing invalid outputs.

How to control

- In code, toggle the behavior via your app config, e.g.:

  app_cfg.ocr.fail_on_merge_error = False  # allow fallback to raw OCR regions

- Default behaviour is to fail-fast to surface postprocessing problems.

Tests added

- `tests/test_merge_overlapping_regions_and_failfast.py` contains:
  - `test_merge_overlapping_regions_simple` — unit test for merging overlapping boxes
  - `test_detect_text_regions_failfast_on_merge` — ensures detect_text_regions re-raises when merge fails and `fail_on_merge_error=True`.

Files changed

- `src/ocr/text_detector.py` — merged helper restored and detect_text_regions updated to re-raise on merge errors when configured.
- `src/core/config.py` — added `OCRConfig.fail_on_merge_error: bool = True`.
- `tests/test_merge_overlapping_regions_and_failfast.py` — new unit tests.

Notes & next steps

- Consider expanding fail-fast coverage to other critical stages (PHI mapping, mask creation). I can add a short PR that enumerates broad `except Exception` blocks and proposes targeted changes.
- I ran the new tests locally; they passed (2 passed, 15 warnings). If you'd like, I can run the full test suite and an end-to-end run on your sample PDF next.

