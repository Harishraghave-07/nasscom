import os
from pathlib import Path

import pytest


def _imports_available():
    try:
        import numpy as np  # noqa: F401
        from PIL import Image  # noqa: F401
        from src.core.pipeline import ClinicalImageMaskingPipeline  # noqa: F401
        from src.core.config import AppConfig  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _imports_available(), reason="Pillow, numpy or pipeline imports unavailable")
def test_visual_regression_on_golden_image(tmp_path: Path) -> None:
    """Run the golden input through the pipeline and compare pixels to the saved golden output.

    This test is intentionally conservative: it will skip if the required
    test assets are missing. It asserts that pixel-level differences are
    below 1% to allow minor anti-aliasing changes.
    """
    import numpy as np
    from PIL import Image
    from src.core.pipeline import ClinicalImageMaskingPipeline
    from src.core.config import AppConfig

    assets_dir = Path(__file__).parent / "assets"
    golden_input = assets_dir / "golden_input.png"
    golden_masked = assets_dir / "golden_masked.png"

    if not golden_input.exists() or not golden_masked.exists():
        pytest.skip(f"Golden assets not present under {assets_dir} - create {golden_input} and {golden_masked}")

    # Prepare config that writes temp files under tmp_path
    config = AppConfig()
    config.processing.temp_dir = str(tmp_path / "temp")

    pipeline = ClinicalImageMaskingPipeline(config)

    out_file = tmp_path / "output_masked.png"

    # Run pipeline (may use fallback mocks if heavy deps missing)
    report = pipeline.process_single_image(str(golden_input), output_path=str(out_file))

    assert out_file.exists(), f"Pipeline did not produce output image. Report: {report}"

    # Load images and compare
    expected = Image.open(golden_masked).convert("RGB")
    actual = Image.open(out_file).convert("RGB")

    if expected.size != actual.size:
        pytest.fail(f"Golden and actual image sizes differ: expected={expected.size} actual={actual.size}")

    a = np.array(expected, dtype=np.int16)
    b = np.array(actual, dtype=np.int16)

    diff = np.abs(a - b)
    # consider a pixel different if any channel differs
    per_pixel_diff = np.any(diff > 0, axis=2)
    differing = int(np.count_nonzero(per_pixel_diff))
    total = per_pixel_diff.shape[0] * per_pixel_diff.shape[1]
    percent = (differing / total) * 100.0

    # Allow up to 1% pixel difference to tolerate anti-aliasing/compression
    tol_percent = 1.0
    assert percent <= tol_percent, f"Visual regression exceeded tolerance: {percent:.4f}% > {tol_percent}% (diff pixels={differing} total={total})"
