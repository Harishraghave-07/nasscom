import pytest

from src.ocr.text_detector import TextDetector
from src.core.config import OCRConfig


def test_merge_two_fully_overlapping_boxes():
    cfg = OCRConfig()
    td = TextDetector(cfg)
    regions = [
        {"bbox": [10, 10, 50, 50], "text": "A", "confidence": 0.9},
        {"bbox": [12, 12, 48, 48], "text": "B", "confidence": 0.8},
    ]
    merged = td.merge_overlapping_regions(regions, overlap_thresh=0.5)
    assert len(merged) == 1
    m = merged[0]
    assert m["bbox"] == (10, 10, 50, 50)
    # highest-confidence text chosen
    assert m["text"] in ("A", "B")


def test_merge_partial_overlap_threshold():
    cfg = OCRConfig()
    td = TextDetector(cfg)
    regions = [
        {"bbox": [0, 0, 100, 10], "text": "left", "confidence": 0.9},
        {"bbox": [90, 0, 180, 10], "text": "right", "confidence": 0.85},
    ]
    # small overlap: intersection = 10*10=100, min area = 100*10=1000 -> iom=0.1
    merged_low = td.merge_overlapping_regions(regions, overlap_thresh=0.2)
    assert len(merged_low) == 2
    merged_high = td.merge_overlapping_regions(regions, overlap_thresh=0.05)
    assert len(merged_high) == 1


def test_non_overlapping_boxes_remain():
    cfg = OCRConfig()
    td = TextDetector(cfg)
    regions = [
        {"bbox": [0, 0, 10, 10], "text": "a", "confidence": 0.7},
        {"bbox": [50, 50, 60, 60], "text": "b", "confidence": 0.8},
    ]
    merged = td.merge_overlapping_regions(regions, overlap_thresh=0.2)
    assert len(merged) == 2


def test_empty_and_single_return_as_is():
    cfg = OCRConfig()
    td = TextDetector(cfg)
    assert td.merge_overlapping_regions([]) == []
    single = [{"bbox": [1, 1, 2, 2], "text": "x", "confidence": 0.5}]
    merged = td.merge_overlapping_regions(single)
    assert len(merged) == 1
    assert merged[0]["bbox"] == (1, 1, 2, 2)
