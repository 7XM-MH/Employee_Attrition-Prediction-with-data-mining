"""General utility functions."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT
