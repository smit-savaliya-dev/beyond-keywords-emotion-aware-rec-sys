"""Tests for src/behaviour_charts.py: every chart type writes a PNG; small fakes, no data files."""
import numpy as np

from src import behaviour_charts as bc


def test_every_chart_type_writes_a_png(tmp_path):
    p = [bc.bars(tmp_path / "a.png", "t", "c", ["x", "y"], [0.01, -0.02], "gain", low=[0.0, -0.03], high=[0.02, -0.01]),
         bc.bars(tmp_path / "b.png", "t", "c", ["x", "y"], [0.6, 0.7], "auc", horizontal=True, zero=False, hline=0.5, hline_label="random"),
         bc.grouped(tmp_path / "c.png", "t", "c", ["g1", "g2"], [("A", [0.6, 0.62], None, None, bc.BLUE), ("B", [0.61, 0.63], [0.6, 0.62], [0.62, 0.64], bc.ORANGE)], "auc"),
         bc.lines(tmp_path / "d.png", "t", "c", [5, 10], [("A", [0.6, 0.61], [0.59, 0.6], [0.61, 0.62], bc.BLUE, "o")], "window", "auc"),
         bc.histogram(tmp_path / "e.png", "t", "c", np.random.default_rng(0).random(100), "x", cuts=[0.3, 0.6], cut_labels=["a", "b"]),
         bc.heatmap(tmp_path / "f.png", "t", "c", np.eye(2), ["r1", "r2"], ["c1", "c2"], "x", "y", center=0.5),
         bc.scatter_map(tmp_path / "g.png", "t", "c", np.arange(10.0), np.arange(10.0), np.r_[np.zeros(5), np.ones(5)].astype(int), ["p0", "p1"], "x", "y")]
    sheet = bc.contact_sheet(p[:4], tmp_path / "sheet.png", "overview", cols=2)
    assert all(x.stat().st_size > 1000 for x in [*p, sheet])
