import pytest
from src.ocr.text_detector import TextDetector
from src.core.config import OCRConfig
import numpy as np


def test_word_offset_integrity(monkeypatch, tmp_path):
    # Prepare a minimal image (blank) and a detector with a fake reader
    cfg = OCRConfig()
    td = TextDetector(cfg)

    # Mock reader.readtext to return two words: 'John' and 'Doe'
    def fake_readtext(crop, detail=1, paragraph=False):
        # bbox points relative to crop; confidence arbitrary
        return [
            ([(0, 0), (40, 0), (40, 10), (0, 10)], 'John', 0.9),
            ([(50, 0), (90, 0), (90, 10), (50, 10)], 'Doe', 0.9),
        ]

    # Install fake reader
    class FakeReader:
        def readtext(self, crop, detail=1, paragraph=False):
            return fake_readtext(crop, detail=detail, paragraph=paragraph)

    td.reader = FakeReader()
    # Construct a fake region that covers the full image
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    region = {"bbox": (0, 0, 200, 100), "confidence": 0.9}

    extracted = td.extract_text_content(img, [region])
    assert len(extracted) == 1
    r = extracted[0]
    assert r["text"] == "John Doe"
    wb = r["word_boxes"]
    assert wb[0]["text"] == "John"
    assert wb[0]["start_char"] == 0
    assert wb[0]["end_char"] == 4
    assert wb[1]["text"] == "Doe"
    # 'John' (4) + space (1) -> start 5
    assert wb[1]["start_char"] == 5
    assert wb[1]["end_char"] == 8
