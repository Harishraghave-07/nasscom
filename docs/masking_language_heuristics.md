Masking language heuristics

Overview
--------
The ImageInpainter now supports coarse language-aware heuristics to choose surgical padding and inpainting radius. This helps avoid over-masking for Latin scripts while using larger padding and stronger inpainting for dense scripts such as CJK.

Configuration
-------------
The following fields were added to `MaskingConfig`:

- `lang_padding_default`: default surgical padding in pixels
- `lang_padding_cjk`: surgical padding for CJK or dense fonts
- `inpainting_radius_cjk`: inpainting radius tuned for CJK scripts

These can be set via code, environment variables (if using the AppConfig), or a `.env` file.

How it works
------------
- Language detection: coarse detection is applied per-page using `detect_language_group()` which checks the configured OCR `supported_languages` and/or inspects detection text for CJK Unicode ranges.
- Heuristics: `get_language_heuristics()` returns `padding` and `inpainting_radius` values for the detected language group.
- Usage: `unified_masking_stage()` and `apply_surgical_masking()` use these heuristics automatically. The metadata returned by `unified_masking_stage()` includes `lang_group`, `surgical_padding_used`, and `inpainting_radius_used` for auditing and testability.

Extending
---------
To add heuristics for additional scripts or font density detection:
1. Extend `detect_language_group()` to recognize the new group.
2. Add new config fields in `MaskingConfig` (for example, `lang_padding_arabic`).
3. Update `get_language_heuristics()` to return the appropriate parameters for the new group.
4. Optionally, refine heuristics using OCR token-level font size or stroke-width estimators if the OCR exposes that data.

Testing & Visual outputs
------------------------
Unit tests were added under `tests/test_language_heuristics_masking.py` to validate that surgical padding and inpainting radii are chosen per language group. Visual comparison scripts can be built in CI to generate side-by-side images for manual review.

Notes
-----
This approach uses coarse heuristics and is conservative by design. For production-grade multilingual documents, consider integrating font detection or OCR model metadata for more precise heuristics.
