import pytest

from src.core.pipeline import ClinicalImageMaskingPipeline
from src.core.config import AppConfig


def make_region(text, word_boxes):
    return {"text": text, "bbox": [0, 0, 100, 10], "word_boxes": word_boxes}


def test_remap_multiple_replacements():
    cfg = AppConfig()
    pipeline = ClinicalImageMaskingPipeline(cfg)

    # original page text contains two sensitive spans
    page_text = "Alice Smith visited Bob Johnson on 2020-01-01"
    # anonymizer replaces both names with [NAME]
    anonymized = "[NAME] visited [NAME] on 2020-01-01"

    regions = [
        make_region("Alice Smith", [{"text": "Alice", "bbox": [0,0,40,10], "start_char": 0, "end_char": 5}, {"text":"Smith","bbox":[41,0,80,10], "start_char":6, "end_char":11}]),
        make_region("Bob Johnson", [{"text":"Bob","bbox":[90,0,120,10], "start_char":18, "end_char":21}, {"text":"Johnson","bbox":[121,0,180,10], "start_char":22, "end_char":29}]),
    ]

    detections = [
        {"text": "Alice Smith", "start": 0, "end": 11, "bbox": [0,0,80,10]},
        {"text": "Bob Johnson", "start": 18, "end": 29, "bbox": [90,0,180,10]},
    ]

    remapped = pipeline._remap_offsets_after_anonymize(page_text, anonymized, regions, detections)
    # starts should point to either 0 or later positions for anonymized placeholders
    assert remapped[0]["start"] >= 0
    assert remapped[1]["start"] >= 0
    # bbox should still be present
    assert "bbox" in remapped[0]


def test_remap_insertion_and_deletion():
    cfg = AppConfig()
    pipeline = ClinicalImageMaskingPipeline(cfg)

    page_text = "ID: 12345 - Note: Patient X"
    anonymized = "ID: [REDACTED] - Note: Patient"

    regions = [make_region("12345", [{"text":"12345","bbox":[4,0,54,10],"start_char":4,"end_char":9}])]
    detections = [{"text":"12345","start":4,"end":9,"bbox":[4,0,54,10]}]

    remapped = pipeline._remap_offsets_after_anonymize(page_text, anonymized, regions, detections)
    assert remapped[0]["start"] >= 0
    assert remapped[0]["end"] >= remapped[0]["start"]
