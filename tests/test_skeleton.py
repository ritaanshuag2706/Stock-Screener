"""Stage 0 smoke test: the package is importable and paths resolve sanely."""

from pathlib import Path

import nse_screener
from nse_screener import paths


def test_package_imports():
    assert nse_screener.__version__


def test_subpackages_import():
    import importlib

    for name in ("data", "patterns", "study", "backtest", "apps"):
        importlib.import_module(f"nse_screener.{name}")


def test_data_paths_sit_under_data_dir():
    assert paths.RAW_DIR.parent == paths.DATA_DIR
    assert paths.BARS_DIR.parent == paths.DATA_DIR
    assert paths.CONFIG_DIR.name == "config"


def test_repo_root_contains_pyproject():
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()
    assert isinstance(paths.REPO_ROOT, Path)
