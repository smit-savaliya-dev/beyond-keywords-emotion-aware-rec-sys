"""scripts/06_present_charts.py runs on the saved tables and writes every chart, its CSV and the README (into a temp folder)."""
import importlib
import sys

import pandas as pd

from src.config import ROOT


def test_present_charts_write_all_files(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    mod = importlib.import_module("06_present_charts")
    data, skipped = mod.main(tmp_path)
    assert skipped == []
    for name, _, _ in mod.CHARTS:
        assert (tmp_path / name).stat().st_size > 10_000
        assert not pd.read_csv(tmp_path / name.replace(".png", ".csv")).empty
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert all(name in text for name, _, _ in mod.CHARTS) and "Numbers to quote" in text
    six = data["06_emotion_gain_vs_twin.png"]
    assert (six["setting"].str.contains("previous days only").sum(), six["setting"].str.contains("with same-day clicks").sum()) == (1, 1)
