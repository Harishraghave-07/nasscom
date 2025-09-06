import sys
from pathlib import Path


def pytest_configure(config):
    """Ensure the project root (one level above tests/) is on sys.path so
    imports like `from src.core.config import AppConfig` work during tests.
    """
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
