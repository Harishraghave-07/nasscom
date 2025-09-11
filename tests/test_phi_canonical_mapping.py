import pytest

from src.ocr.mapping import map_text_spans_to_image_bboxes


def test_map_text_spans_to_image_bboxes_basic():
    # one word box with char spans
    word_boxes = [{"text": "John", "bbox": [10, 10, 50, 30], "start_char": 0, "end_char": 4}]
    detections = [{"text": "John", "start": 0, "end": 4}]
    mapped = map_text_spans_to_image_bboxes(detections, word_boxes, region_bbox=[10, 10, 50, 30], page_number=1)
    assert len(mapped) == 1
    assert mapped[0]["bbox"] == [10, 10, 50, 30]
    assert mapped[0]["page_number"] == 1


def test_map_text_spans_fallback_text_match():
    word_boxes = [
        {"text": "John", "bbox": [10, 10, 50, 30]},
        {"text": "Doe", "bbox": [60, 10, 90, 30]},
    ]
    detections = [{"text": "Doe", "start": None, "end": None}]
    mapped = map_text_spans_to_image_bboxes(detections, word_boxes, region_bbox=[0, 0, 100, 100], page_number=2)
    assert len(mapped) == 1
    assert mapped[0]["bbox"] == [60, 10, 90, 30]
    assert mapped[0]["page_number"] == 2
