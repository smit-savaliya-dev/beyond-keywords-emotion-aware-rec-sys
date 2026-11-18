"""Final multi-metric tables (scripts/08_03_final_metrics.py, src/final_eval.py) and the final chart set (scripts/07_final_figures.py)."""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from src.config import ROOT, TABLES
from src.evaluate import evaluate
from src.final_eval import RANKING, evaluate_with_lists, twin_gains


def fake(n_imp=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_imp):
        k = int(rng.integers(3, 15))
        lab = np.zeros(k, int)
        lab[rng.choice(k, int(rng.integers(1, k)), replace=False)] = 1
        for j in range(k):
            rows.append({"imp_key": f"i{i:03d}", "news_id": f"N{int(rng.integers(0, 30))}_{i}_{j}", "label": lab[j], "position": j})
    truth = pd.DataFrame(rows)
    scores = truth[["imp_key", "news_id"]].assign(score=np.round(rng.random(len(truth)), 1))    # many ties on purpose
    cats = pd.Series({n: f"c{hash(n) % 4}" for n in truth["news_id"]})
    emos = pd.Series({n: f"e{hash(n) % 3}" for n in truth["news_id"]})
    return truth, scores, cats, emos


def test_metrics_equal_the_main_evaluation_with_ties():
    truth, scores, cats, emos = fake()
    main, _ = evaluate(truth, scores)
    mine, shown = evaluate_with_lists(truth, scores, cats, emos)
    for k in RANKING:
        assert np.allclose(main[k], mine[k], equal_nan=True), k
    assert (mine["cat_div@10"] <= 4).all() and (mine["emo_div@10"] <= 3).all() and (mine["cat_div@10"] >= 1).all()
    assert shown <= set(truth["news_id"])


def test_top10_is_the_main_tie_break_order():
    """The top 10 used for diversity follows the same order that gives the main P@10 (hits in the top 10)."""
    truth, scores, cats, emos = fake(seed=3)
    lab = pd.Series(truth["label"].map({1: "click", 0: "none"}).to_numpy(), index=truth["news_id"])
    per, _ = evaluate_with_lists(truth, scores, lab, emos)
    main, _ = evaluate(truth, scores)
    has_click_and_none = per["cat_div@10"] == 2
    only_click = (main["p@10"] * 10 == truth.groupby("imp_key").size().clip(upper=10).to_numpy())
    assert ((per["cat_div@10"] == 1) == (only_click | (main["p@10"] == 0))).all() and has_click_and_none.any()


def test_twin_gains_holm_over_all_tests():
    truth, scores, cats, emos = fake()
    a, _ = evaluate_with_lists(truth, scores, cats, emos)
    b, _ = evaluate_with_lists(truth, scores.assign(score=scores["score"] + truth["label"] * 0.05), cats, emos)
    users = pd.Series(["u" + k[-1] for k in a["imp_key"]], index=a["imp_key"])
    g = twin_gains([("pair", "A", "B")], {"A": a, "B": b}, users, {"A": 0.5, "B": 0.6})
    assert len(g) == 11 and g.loc[g["metric"] == "auc", "gain"].iloc[0] > 0
    assert g.loc[g["metric"] == "coverage@10", "gain"].iloc[0] == pytest.approx(0.1) and not g.loc[g["metric"] == "coverage@10", "tested"].iloc[0]


def test_final_auc_equals_the_existing_final_tables():
    m = pd.read_csv(TABLES / "final_all_metrics.csv")
    s = pd.read_csv(TABLES / "summary.csv")
    s = s[s["part"] == "dev"].set_index("model")["auc"]
    checked = 0
    for _, r in m.iterrows():
        if r["model"] in s.index:
            assert round(r["auc"], 4) == pytest.approx(s[r["model"]], abs=1e-9), r["model"]
            checked += 1
    assert checked == 15
    ladder = pd.read_csv(TABLES / "model_ladder.csv").set_index("model")["auc"]
    for code in ("M2_sbert", "M3_sbert_emotion", "M6_nrms", "M7b_nrms_emotion_gated"):
        assert m.set_index("model").loc[code, "auc"] == pytest.approx(ladder[code], abs=1e-12)


def test_final_figures_write_every_file(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    mod = importlib.import_module("07_final_figures")
    data, skipped = mod.main(tmp_path)
    assert skipped == []
    for folder, name, _, _ in mod.CHARTS:
        base = tmp_path / folder / name if folder else tmp_path / name
        assert base.with_suffix(".png").stat().st_size > 10_000 and not pd.read_csv(base.with_suffix(".csv")).empty
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Names used in charts" in text and all(f"{n}.png" in text for _, n, _, _ in mod.CHARTS)


def test_week9_metrics_auc_equals_the_final_results_tables():
    """final_metrics_all_models.csv (scripts/08_04_week9_metrics.py): AUC equals the main summary (dev) for every saved model, the
    same-weight blend equals the earlier final table, and the gains table repeats the two model values."""
    m = pd.read_csv(TABLES / "final_metrics_all_models.csv").set_index("model")
    s = pd.read_csv(TABLES / "summary.csv")
    s = s[s["part"] == "dev"].set_index("model")
    for code in m.index.drop("B1_R1_same_weight"):
        for k in RANKING:
            assert round(m.loc[code, k], 4) == pytest.approx(s.loc[code, k], abs=1e-9), (code, k)
    old = pd.read_csv(TABLES / "final_all_metrics.csv").set_index("model")
    assert m.loc["B1_R1_same_weight", "auc"] == pytest.approx(old.loc["B1_R1_same_weight", "auc"], abs=1e-12)
    g = pd.read_csv(TABLES / "final_emotion_gains_all_metrics.csv")
    assert len(g) == 5 * 8 and (g["holm_family"].str.startswith("standard")).sum() == 32
    for _, r in g.iterrows():
        assert r["twin_value"] == pytest.approx(m.loc[r["twin"], r["metric"]], abs=1e-12)
        assert r["gain"] == pytest.approx(m.loc[r["emotion_model"], r["metric"]] - m.loc[r["twin"], r["metric"]], abs=1e-12)
        assert r["ci_low"] <= r["gain"] <= r["ci_high"]
    assert not m["display_name"].str.contains(r"\bM\d|v6|_", regex=True).any()


def test_week9_figures_write_every_file_with_names_not_codes(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    mod = importlib.import_module("08_week9_figures")
    created = mod.main(tmp_path)
    assert len(created) == 13
    for folder, name, takeaway, _ in created:
        base = (tmp_path / folder / name) if folder else (tmp_path / name)
        assert base.with_suffix(".png").stat().st_size > 10_000 and takeaway
        d = pd.read_csv(base.with_suffix(".csv"))
        assert not d.empty
        for col in ("display_name", "model", "label"):
            if col in d:
                assert not d[col].astype(str).str.contains(r"\bM\d|v6|_R1|nrms_", regex=True).any(), (name, col)
        assert (tmp_path / "week9" / f"{name}.png").exists()
    assert not list((tmp_path / "week9").glob("*.csv"))
    text = (tmp_path / "week9" / "README.md").read_text(encoding="utf-8")
    assert "## Names used" in text and "## Metric glossary" in text and "M7d_nrms_emotion_fusion" in text
    assert "<!-- week9:start -->" in (tmp_path / "README.md").read_text(encoding="utf-8")
