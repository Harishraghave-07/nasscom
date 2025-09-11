import numpy as np
import pytest
from src.masking.image_inpainter import ImageInpainter
from src.core.config import MaskingConfig


def make_dummy_image():
    # white background image
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


def test_merge_entity_masks_union_bbox():
    img = make_dummy_image()
    # define two word boxes representing two tokens of same entity
    word_boxes = [
        {"bbox": [10, 10, 20, 20], "text": "A"},
        {"bbox": [30, 30, 40, 40], "text": "B"},
    ]
    region = {"bbox": [10, 10, 40, 40], "text": "AB", "word_boxes": word_boxes}
    inp = ImageInpainter(MaskingConfig())
    # no padding, merge_entity=True should mask union bbox [10,10] to [40,40]
    masked = inp.apply_surgical_masking(img, [region], padding=0, merge_entity=True)
    # count masked pixels (non-white)
    mask_area = np.sum(np.any(masked != 255, axis=2))
    expected_area = (40 - 10) * (40 - 10)
    assert mask_area == expected_area, f"Expected masked area {expected_area}, got {mask_area}"


def test_no_merge_entity_masks_individual_tokens():
    img = make_dummy_image()
    word_boxes = [
        {"bbox": [10, 10, 20, 20], "text": "A"},
        {"bbox": [30, 30, 40, 40], "text": "B"},
    ]
    region = {"bbox": [10, 10, 40, 40], "text": "AB", "word_boxes": word_boxes}
    inp = ImageInpainter(MaskingConfig())
    masked = inp.apply_surgical_masking(img, [region], padding=0, merge_entity=False)
    mask_area = np.sum(np.any(masked != 255, axis=2))
    # two 10x10 squares
    expected_area = (20 - 10) * (20 - 10) + (40 - 30) * (40 - 30)
    assert mask_area == expected_area, f"Expected masked area {expected_area}, got {mask_area}"


def test_padding_respected():
    img = make_dummy_image()
    word_boxes = [{"bbox": [50, 50, 60, 60], "text": "C"}]
    region = {"bbox": [50, 50, 60, 60], "text": "C", "word_boxes": word_boxes}
    inp = ImageInpainter(MaskingConfig())
    # with padding=2, union bbox should expand by 2 on all sides
    masked = inp.apply_surgical_masking(img, [region], padding=2, merge_entity=True)
    # masked area should be (width+2*pad)^2 = (10+4)^2
    mask_area = np.sum(np.any(masked != 255, axis=2))
    expected_area = (60 - 50 + 4) * (60 - 50 + 4)
    assert mask_area == expected_area, f"Expected masked area {expected_area}, got {mask_area}"
