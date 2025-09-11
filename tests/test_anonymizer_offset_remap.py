import pytest

from src.core.pipeline import ClinicalImageMaskingPipeline
from src.core.config import AppConfig


def test_remap_offsets_after_anonymize_simple():
    cfg = AppConfig()
    pipeline = ClinicalImageMaskingPipeline(cfg)

    page_text = "John Doe went home"
    anonymized_text = "[NAME] went home"
    # regions: one region containing John Doe
    regions = [{"text": "John Doe", "bbox": [0, 0, 100, 10], "word_boxes": [
        {"text": "John", "bbox": [0,0,50,10], "start_char": 0, "end_char": 4},
        {"text": "Doe", "bbox": [51,0,100,10], "start_char": 5, "end_char": 8}
    ]}]

    detections = [{"text": "John Doe", "start": 0, "end": 8, "bbox": [0,0,100,10], "confidence": 0.9}]

    remapped = pipeline._remap_offsets_after_anonymize(page_text, anonymized_text, regions, detections)
    # Ensure start/end updated and bbox present
    assert remapped[0]["start"] >= 0
    assert remapped[0]["end"] >= remapped[0]["start"]
    assert "bbox" in remapped[0]


def test_compute_orig_to_new_index_map_identity():
    cfg = AppConfig()
    pipeline = ClinicalImageMaskingPipeline(cfg)
    orig = "hello world"
    new = "hello world"
    m = pipeline._compute_orig_to_new_index_map(orig, new)
    assert len(m) == len(orig) + 1
    # each index should map to itself
    for i in range(len(orig) + 1):
        assert m[i] == i
