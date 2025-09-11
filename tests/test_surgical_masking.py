import os
import pytest
from src.core.config import AppConfig
from src.core.pipeline import ClinicalImageMaskingPipeline


def test_surgical_masking():
    # Create a test config with surgical masking enabled
    cfg = AppConfig()
    cfg.mask.redaction_style = "surgical"
    cfg.mask.surgical_padding_pixels = 3

    pipeline = ClinicalImageMaskingPipeline(cfg)

    # Use a small test PDF/PNG in the repo if available; otherwise skip
    sample = os.path.join(os.path.dirname(__file__), "fixtures", "sample_phi_page.png")
    if not os.path.exists(sample):
        pytest.skip("No sample fixture available for surgical masking test")

    result = pipeline.process_single_image(sample)

    # Ensure phi_regions present in the report
    phi = result.get("phi_regions") or []
    assert len(phi) > 0, "No PHI regions detected"

    # Check that regions are reasonably small
    for region in phi:
        bbox = region.get("bbox") or region.get("box")
        assert bbox and len(bbox) == 4
        x1, y1, x2, y2 = map(int, bbox)
        width = x2 - x1
        height = y2 - y1
        assert width < 200, f"Region too wide: {width}px (expected <200px)"
        assert height < 50, f"Region too tall: {height}px (expected <50px)"

    print(f"\u2713 Surgical masking test passed: {len(phi)} precise regions")
    return result
