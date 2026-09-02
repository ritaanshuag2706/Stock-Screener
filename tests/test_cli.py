"""The CLI entry points actually run.

Added after a column rename broke `apps.scan` while every unit test still
passed -- nothing had ever executed the command end to end. These are smoke
tests: they assert the thing runs and prints, not what the numbers are.
"""

import pandas as pd
import pytest

from nse_screener.apps import scan as scan_cli
from nse_screener.apps import show as show_cli
from nse_screener.data import store
from tests.test_screener import HAMMER, make_bars


@pytest.fixture
def populated(tmp_path, monkeypatch):
    """A store with one eligible symbol, wired in as the default location."""
    bars_dir = tmp_path / "bars"
    store.append(make_bars("AAA", n=400, last=HAMMER), bars_dir=bars_dir)
    monkeypatch.setattr(store, "BARS_DIR", bars_dir)
    return bars_dir


def test_scan_cli_runs(populated, capsys):
    assert scan_cli.main([]) == 0
    out = capsys.readouterr().out
    assert "AAA" in out
    assert "Hammer" in out          # display label, not the identifier


def test_scan_cli_prints_every_declared_column(populated, capsys):
    scan_cli.main([])
    header = capsys.readouterr().out.splitlines()[3]
    for col in ("Date", "Symbol", "Pattern", "Close", "Chg", "Volume",
                "Trend", "vs200", "vs25", "RelVol", "RSI"):
        assert col in header, f"{col} missing from the CLI table"


def test_scan_cli_multi_session(populated, capsys):
    assert scan_cli.main(["--days", "3"]) == 0
    assert "session(s)" in capsys.readouterr().out


def test_scan_cli_rejects_an_unknown_pattern(populated, capsys):
    assert scan_cli.main(["--patterns", "marubozu"]) == 2
    assert "unknown pattern" in capsys.readouterr().err


def test_scan_cli_writes_csv(populated, tmp_path, capsys):
    dest = tmp_path / "hits.csv"
    assert scan_cli.main(["--csv", str(dest)]) == 0
    written = pd.read_csv(dest)
    assert "symbol" in written.columns and len(written) >= 1


def test_show_cli_runs(populated, capsys):
    assert show_cli.main(["AAA", "--last", "5"]) == 0
    assert "AAA" in capsys.readouterr().out


def test_show_cli_unknown_symbol(populated, capsys):
    assert show_cli.main(["NOSUCH"]) == 1
    assert "no bars stored" in capsys.readouterr().err
