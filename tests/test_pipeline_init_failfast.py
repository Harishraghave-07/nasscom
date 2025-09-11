import pytest

from src.core.config import AppConfig


def test_init_components_failfast(monkeypatch):
    # Force an import-time error in one of the components to verify fail-fast behaviour
    import importlib

    from src.core.pipeline import ClinicalImageMaskingPipeline

    # monkeypatch the ImageProcessor class import path to raise
    def bad_import(*args, **kwargs):
        raise RuntimeError("simulated init failure")

    monkeypatch.setattr('src.preprocessing.image_processor.ImageProcessor', bad_import, raising=False)

    cfg = AppConfig()
    with pytest.raises(RuntimeError):
        ClinicalImageMaskingPipeline(cfg)
