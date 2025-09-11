import pytest
from src.ocr.text_detector import merge_adjacent_regions


def make_word(text, bbox, conf=0.9):
    return {"text": text, "bbox": bbox, "confidence": conf}


def test_merge_simple_multiword_phi():
    # two adjacent words forming a PHI phrase "John Doe"
    regs = [
        make_word("John", [10, 10, 60, 30]),
        make_word("Doe", [62, 10, 110, 30]),
        make_word("Age", [200, 10, 240, 30]),
    ]
    merged = merge_adjacent_regions(regs, ["John Doe"], fuzz_threshold=80)
    # expect merged region for John Doe plus Age
    texts = [r["text"] for r in merged]
    assert any(t.lower() == "john doe" for t in texts)
    assert any(t.lower() == "age" for t in texts)


def test_no_merge_non_adjacent():
    regs = [
        make_word("John", [10, 10, 60, 30]),
        make_word("Random", [300, 10, 380, 30]),
        make_word("Doe", [400, 10, 460, 30]),
    ]
    merged = merge_adjacent_regions(regs, ["John Doe"], fuzz_threshold=80)
    # should not merge because John and Doe are far apart
    texts = [r["text"] for r in merged]
    assert "John" in texts
    assert "Doe" in texts


def test_merge_with_minor_ocr_errors():
    # simulate OCR error: "D0e" instead of "Doe"
    regs = [
        make_word("Emily", [10, 10, 70, 30]),
        make_word("D0e", [72, 10, 120, 30]),
    ]
    merged = merge_adjacent_regions(regs, ["Emily Doe"], fuzz_threshold=70)
    texts = [r["text"].lower() for r in merged]
    assert any("emily" in t and "doe" in t.replace("0", "o") or "d0e" in t for t in texts)


if __name__ == "__main__":
    pytest.main([__file__])
