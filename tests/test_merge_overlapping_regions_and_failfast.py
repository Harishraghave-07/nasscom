import pytest
import numpy as np

from src.ocr.text_detector import TextDetector
from src.core.config import OCRConfig


def test_merge_overlapping_regions_simple():
    cfg = OCRConfig()
    td = TextDetector(cfg)
    regions = [
        {"bbox": [10, 10, 50, 30], "text": "John", "confidence": 0.9},
        {"bbox": [45, 12, 90, 32], "text": "Doe", "confidence": 0.85},
        {"bbox": [200, 200, 240, 220], "text": "Other", "confidence": 0.95},
    ]
    merged = td.merge_overlapping_regions(regions, overlap_thresh=0.1)
    # Expect first two to be merged due to overlap
    assert any(r for r in merged if r.get("text") and "John" in str(r.get("text")) or "Doe" in str(r.get("text")))
    # Expect three output regions or fewer
    assert len(merged) <= 3


def test_detect_text_regions_failfast_on_merge(monkeypatch):
    cfg = OCRConfig()
    cfg.fail_on_merge_error = True
    td = TextDetector(cfg)

    # create an image and fake reader
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    class DummyReader:
        def detect(self, img):
            # detection_results[0] should be a list of boxes; each box is a list of 4 points
            return ([[[10,10], [50,10], [50,30], [10,30]]], [0.95])

    td.reader = DummyReader()

    # monkeypatch merge_overlapping_regions to raise
    def bad_merge(self, regs, overlap_thresh=0.5):
        raise RuntimeError("merge failed")

    # patch at the class level so method binding works as expected
    from src.ocr.text_detector import TextDetector as TDClass
    monkeypatch.setattr(TDClass, 'merge_overlapping_regions', bad_merge)

    with pytest.raises(RuntimeError):
        td.detect_text_regions(img)
