import numpy as np
from src.core.config import AppConfig
from src.core.pipeline import ClinicalImageMaskingPipeline


def make_region(text, bbox, word_boxes):
    return {"text": text, "bbox": bbox, "word_boxes": word_boxes}


def test_remap_after_anonymize_simple_replace():
    cfg = AppConfig()
    pipeline = ClinicalImageMaskingPipeline(cfg)

    # Build simple page: two regions joined by a space
    # Region A: 'Name: Alice' with word token boxes
    region_a = make_region(
        "Name: Alice",
        [10, 10, 200, 40],
        [
            {"text": "Name:", "bbox": [10, 10, 80, 40], "start_char": 0, "end_char": 5},
            {"text": "Alice", "bbox": [85, 10, 200, 40], "start_char": 6, "end_char": 11},
        ],
    )

    # Region B: some other text
    region_b = make_region("Age: 30", [10, 50, 120, 80], [{"text": "Age:", "bbox": [10, 50, 50, 80], "start_char": 0, "end_char": 4}, {"text": "30", "bbox": [55, 50, 70, 80], "start_char": 5, "end_char": 7}])

    regions = [region_a, region_b]

    # detection: Alice from region A
    detections = [{"text": "Alice", "start": 6, "end": 11, "entity_type": "names", "confidence": 0.99}]

    page_text = " ".join([r["text"] for r in regions])

    # anonymized text replaced 'Alice' with '[REDACTED]' (longer)
    anonymized = page_text.replace("Alice", "[REDACTED]")

    remapped = pipeline._remap_offsets_after_anonymize(page_text, anonymized, regions, detections)

    assert isinstance(remapped, list)
    assert remapped[0].get("bbox") is not None
    # bbox should still be in roughly the same x-range (start near 85..200)
    bx = remapped[0]["bbox"]
    assert bx[0] >= 0 and bx[2] > bx[0]


def test_remap_with_multiple_replacements_and_shrink():
    cfg = AppConfig()
    pipeline = ClinicalImageMaskingPipeline(cfg)

    # page with two names
    r1 = make_region(
        "Patient: Bob",
        [10, 10, 180, 40],
        [{"text": "Patient:", "bbox": [10, 10, 90, 40], "start_char": 0, "end_char": 8}, {"text": "Bob", "bbox": [92, 10, 120, 40], "start_char": 9, "end_char": 12}],
    )
    r2 = make_region(
        "Contact: 999-9999",
        [10, 50, 220, 80],
        [{"text": "Contact:", "bbox": [10, 50, 80, 80], "start_char": 0, "end_char": 8}, {"text": "999-9999", "bbox": [85, 50, 160, 80], "start_char": 9, "end_char": 16}],
    )
    regions = [r1, r2]

    detections = [
        {"text": "Bob", "start": 9, "end": 12, "entity_type": "names", "confidence": 0.95},
        {"text": "999-9999", "start": 9 + len(r1["text"]) + 1, "end": 16 + len(r1["text"]) + 1, "entity_type": "phone_numbers", "confidence": 0.95},
    ]

    page_text = " ".join([r["text"] for r in regions])

    # anonymize Bob -> XX (shorter) and phone -> [NUMBER] (shorter)
    anonymized = page_text.replace("Bob", "XX").replace("999-9999", "[NUMBER]")

    remapped = pipeline._remap_offsets_after_anonymize(page_text, anonymized, regions, detections)

    assert len(remapped) == 2
    for d in remapped:
        assert d.get("bbox") is not None
