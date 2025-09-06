import sys
import types
import urllib.request
import shutil
import logging

import pytest

from src.core.config import AppConfig


def test_disk_space_insufficient(monkeypatch, tmp_path):
    """If disk free space is below threshold, validation should fail."""
    cfg = AppConfig(processing={"temp_dir": str(tmp_path)})

    class FakeUsage:
        total = 100
        used = 99
        free = 500 * 1024 * 1024  # 500 MB

    def fake_disk_usage(path):
        return FakeUsage()

    monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)

    with pytest.raises(ValueError, match="Insufficient disk space"):
        cfg.validate_clinical_environment()


def test_memory_insufficient(monkeypatch, tmp_path):
    """If available memory is too low, validation should fail (psutil present).

    We inject a fake psutil module with virtual_memory() returning small
    available memory to simulate low-memory environments.
    """
    cfg = AppConfig(processing={"temp_dir": str(tmp_path)})

    fake_psutil = types.SimpleNamespace()
    fake_psutil.virtual_memory = lambda: types.SimpleNamespace(available=200 * 1024 * 1024)  # 200MB

    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    with pytest.raises(ValueError, match="Insufficient available memory"):
        cfg.validate_clinical_environment()


def test_network_unavailable_warns(monkeypatch, caplog, tmp_path):
    """Network failure during model download check should warn but not fail."""
    cfg = AppConfig(processing={"temp_dir": str(tmp_path)})

    def fake_urlopen(url, timeout=3):
        raise urllib.request.URLError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    caplog.set_level(logging.WARNING)
    # Should not raise, only warn
    assert cfg.validate_clinical_environment() is True
    assert any("Network check failed" in rec.message for rec in caplog.records)


def test_prod_requires_secret_and_audit(tmp_path):
    """Production environment must have audit trail and secret_key set."""
    proc = {"temp_dir": str(tmp_path)}

    # Missing audit trail
    with pytest.raises(ValueError, match="Audit trail must be enabled"):
        AppConfig(environment="prod", logging={"enable_audit_trail": False}, processing=proc).validate_clinical_environment()

    # Audit enabled but missing secret_key
    with pytest.raises(ValueError, match="secret_key must be set"):
        AppConfig(environment="prod", logging={"enable_audit_trail": True}, processing=proc).validate_clinical_environment()
