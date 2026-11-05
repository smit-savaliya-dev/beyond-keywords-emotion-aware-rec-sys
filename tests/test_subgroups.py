"""Tests for src/subgroups.py (fast metrics, group gains with a user bootstrap, trend tests, shuffled labels), with small fakes:
no internet, GPU or real lexicons."""
import numpy as np
import pandas as pd
import pytest

from src import subgroups as sg
from src.evaluate import evaluate


def fake_truth(n_imp=40, size=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_imp):
        labels = np.zeros(size, dtype=int)
        labels[rng.choice(size, rng.integers(1, 3), replace=False)] = 1
        rows += [{"imp_key": f"i{i:03d}", "news_id": f"n{j}", "label": int(labels[j]), "position": j} for j in range(size)]
    return pd.DataFrame(rows)


def test_fast_metrics_match_the_harness_without_ties():
    truth = fake_truth()
    scores = np.random.default_rng(1).random(len(truth))
    fast = sg.fast_metrics(truth["imp_key"], truth["label"], scores)
    per, _ = evaluate(truth, truth[["imp_key", "news_id"]].assign(score=scores))
    m = fast.merge(per, on="imp_key", suffixes=("_f", "_h"))
    for k in ("auc", "mrr", "ndcg@10"):
        assert np.allclose(m[f"{k}_f"], m[f"{k}_h"]), k


def test_restricted_metrics_use_only_masked_rows():
    truth = fake_truth()
    scores = truth["label"].to_numpy(float)          # an oracle
    out = sg.restricted_metrics(truth, scores, truth["position"].to_numpy() < 3)
    assert ((out["auc"] == 1) | out["auc"].isna()).all()


def test_group_gains_flag_small_groups_and_bootstrap_covers_the_mean():
    truth = fake_truth(200)
    rng = np.random.default_rng(2)
    a = sg.fast_metrics(truth["imp_key"], truth["label"], rng.random(len(truth)))
    b = sg.fast_metrics(truth["imp_key"], truth["label"], truth["label"] + rng.random(len(truth)))
    groups = pd.Series(np.where(np.arange(len(a)) < 150, "big", "small"), index=a["imp_key"])
    users = pd.Series([f"u{i % 50}" for i in range(len(a))], index=a["imp_key"])
    t = sg.group_gains(a, b, groups, users, metrics=("auc",), min_group=100, n_boot=200)
    big, small = t.set_index("group").loc["big"], t.set_index("group").loc["small"]
    assert not big["too_small"] and small["too_small"] and big["impressions"] == 150
    assert big["ci_low"] <= big["gain"] <= big["ci_high"] and big["gain"] > 0


def test_user_bootstrap_ci_is_degenerate_for_a_constant():
    lo, hi = sg.user_bootstrap_ci(np.full(30, 0.2), np.arange(30) % 5, n_boot=50)
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)


def test_trend_tests_see_a_real_trend_and_not_noise():
    rng = np.random.default_rng(3)
    s = rng.random(3000)
    groups = np.where(s < 1 / 3, "low", np.where(s < 2 / 3, "mid", "high"))
    real = sg.trend_tests(s, s + rng.normal(0, 0.3, 3000), groups)
    noise = sg.trend_tests(s, rng.normal(0, 0.3, 3000), groups)
    assert real["rho"] > 0.5 and real["hl_p"] < 1e-6 and abs(noise["rho"]) < 0.1


def test_shuffled_labels_keep_clicks_per_impression():
    truth = fake_truth()
    sh = sg.shuffled_labels(truth)
    assert (sh.groupby("imp_key")["label"].sum() == truth.groupby("imp_key")["label"].sum()).all()
    assert not (sh["label"].to_numpy() == truth["label"].to_numpy()).all()


def test_paired_family_holm_and_user_level():
    truth = fake_truth(300)
    rng = np.random.default_rng(4)
    a = sg.fast_metrics(truth["imp_key"], truth["label"], rng.random(len(truth)))
    good = sg.fast_metrics(truth["imp_key"], truth["label"], truth["label"] * 0.5 + rng.random(len(truth)))
    same = a.copy()
    users = pd.Series([f"u{i % 60}" for i in range(len(a))], index=a["imp_key"])
    t = sg.paired_family([("good", a, good), ("same", a, same)], users).set_index("comparison")
    assert t.loc["good", "significant_holm"] and not t.loc["same", "significant_holm"]
    assert t.loc["good", "users"] == 60 and t.loc["good", "p_holm"] >= t.loc["good", "p"]


def test_within_group_metrics_rank_only_inside_each_group():
    keys = np.array(["i1"] * 4)
    groups = np.array(["news", "news", "sport", "sport"])
    labels = np.array([1, 0, 0, 1])
    t = sg.within_group_metrics(keys, groups, labels, np.array([0.9, 0.1, 0.2, 0.8])).set_index("group")
    assert t.loc["news", "auc"] == 1.0 and t.loc["sport", "auc"] == 1.0 and set(t["imp_key"]) == {"i1"}
