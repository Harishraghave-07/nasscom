from src.ocr.text_detector import TextDetector


def make_det(conf, text="x"):
    return {"text": text, "confidence": conf}


class DummyConfig:
    # Provide the minimal set of attributes used by TextDetector.__init__
    confidence_slack = 0.1
    confidence_threshold = 0.6
    min_text_area = 50
    supported_languages = ["en"]
    gpu_enabled = False


def test_keep_regions_within_slack():
    td = TextDetector(DummyConfig())
    dets = [make_det(0.65, "abc"), make_det(0.55, "longertext")]
    # threshold 0.7, slack 0.1 -> effective threshold for short 'abc' is lower
    accepted, rejected = td.filter_by_confidence(dets, 0.7, slack=0.1)
    assert len(accepted) >= 1


def test_fallback_keep_top():
    td = TextDetector(DummyConfig())
    dets = [make_det(0.2, "a"), make_det(0.1, "bb")]
    accepted, rejected = td.filter_by_confidence(dets, 0.8, slack=0.05)
    assert len(accepted) == 1
    assert accepted[0]["confidence"] == max(d["confidence"] for d in dets)
