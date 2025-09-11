import numpy as np
from src.core.config import MaskingConfig
from src.masking.image_inpainter import ImageInpainter


def make_dummy_image(w=200, h=100, color=255):
    img = np.ones((h, w, 3), dtype=np.uint8) * color
    return img



def test_language_heuristics_latin_defaults():
    cfg = MaskingConfig()
    cfg.lang_padding_default = 3
    cfg.lang_padding_cjk = 10
    cfg.inpainting_radius = 2
    cfg.inpainting_radius_cjk = 6

    inp = ImageInpainter(cfg)
    regions = [{"bbox": [10, 5, 60, 30], "text": "John Doe"}]
    img = make_dummy_image()

    out, meta = inp.unified_masking_stage(img, regions, style="surgical")
    assert meta.get("lang_group") == "latin"
    assert meta.get("surgical_padding_used") == 3

    out2, meta2 = inp.unified_masking_stage(img, regions, style="inpaint")
    assert meta2.get("lang_group") == "latin"
    assert meta2.get("inpainting_radius_used") == 2
