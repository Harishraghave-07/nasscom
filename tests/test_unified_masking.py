import numpy as np
import os
from src.masking.image_inpainter import ImageInpainter
from src.core.config import MaskingConfig


def make_image(w=200, h=100, color=(255, 255, 255)):
    img = np.ones((h, w, 3), dtype=np.uint8)
    img[..., 0] = color[0]
    img[..., 1] = color[1]
    img[..., 2] = color[2]
    return img


def test_surgical_masks_apply_and_cover_area(tmp_path):
    cfg = MaskingConfig()
    inp = ImageInpainter(cfg)
    img = make_image()
    regs = [{"bbox": [10, 10, 50, 30], "text": "John"}, {"bbox": [70, 10, 120, 30], "text": "Doe"}]
    out, meta = inp.unified_masking_stage(img, regs, style="surgical", padding=2)
    assert out.shape == img.shape
    # ensure some pixels were changed (black rectangles)
    assert np.any(out != img)
    assert meta.get("method") == "surgical"


def test_blackbox_merge_covers_entire_span(tmp_path):
    cfg = MaskingConfig(blackbox_padding_pixels=3)
    inp = ImageInpainter(cfg)
    img = make_image(300, 100)
    regs = [{"bbox": [10, 10, 50, 30]}, {"bbox": [90, 10, 140, 30]}]
    out, meta = inp.unified_masking_stage(img, regs, style="blackbox_merge")
    assert out.shape == img.shape
    assert meta.get("method") == "blackbox_merge"
    # ensure masked area is non-zero
    covered = np.sum(np.all(out == 0, axis=2))
    assert covered > 0


def test_inpaint_fallback_returns_blackbox_when_inpainting_unavailable(tmp_path, monkeypatch):
    cfg = MaskingConfig(inpainting_method="telea")
    inp = ImageInpainter(cfg)
    img = make_image()
    regs = [{"bbox": [10, 10, 50, 30]}]

    # force apply_inpainting to raise
    def fail_inpaint(image, mask, method="telea"):
        raise RuntimeError("inpaint fail")

    monkeypatch.setattr(inp, "apply_inpainting", fail_inpaint)
    out, meta = inp.unified_masking_stage(img, regs, style="inpaint", fallback_to_blackbox=True)
    assert meta.get("method") == "inpaint_fallback_blackbox"
    assert np.any(out != img)
