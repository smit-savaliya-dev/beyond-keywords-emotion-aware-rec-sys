"""Tests for src/ltr_v6.py (time-causal window counts, within-impression normalisation, rolling folds) and leakage, with small fakes."""
import numpy as np
import pandas as pd
import pytest

from src import ltr_v6 as lv


def test_window_counts_exact():
    n, s = lv.window_counts([0, 0, 0], [10, 20, 30], [1, 0, 1], [0, 0, 0, 0], [10, 20, 31, 45], width=15)
    assert list(n) == [0, 1, 2, 1] and list(s) == [0, 1, 1, 1]
    n_all, _ = lv.window_counts([0, 0, 0], [10, 20, 30], [1, 0, 1], [0], [30], None)
    assert list(n_all) == [2]                  # the event at the same second is not counted


def test_later_events_never_change_a_row():
    base = lv.window_counts([0, 0], [10, 20], [1, 1], [0], [15], 100)
    more = lv.window_counts([0, 0, 0], [10, 20, 16], [1, 1, 1], [0], [15], 100)
    assert base[0][0] == more[0][0] == 1


def test_within_impression_rank_and_z():
    f = pd.DataFrame({"imp_key": ["a", "a", "a", "b"], "x": [1.0, 2.0, 3.0, 5.0]})
    out = lv.within_impression(f, ["x"])
    assert out["x_rank"].tolist()[:3] == pytest.approx([1 / 3, 2 / 3, 1.0]) and out["x_z"].iloc[3] == 0.0
    assert out["x_z"].iloc[:3].tolist() == pytest.approx([-1.0, 0.0, 1.0])


def test_folds_train_strictly_before_the_test_day():
    days = pd.to_datetime(["2019-11-09", "2019-11-10", "2019-11-11", "2019-11-12", "2019-11-14"])
    tr, te = lv.fold_masks(days, "F2")
    assert list(tr) == [True, True, True, False, False] and list(te) == [False, False, False, True, False]


def test_extra_features_are_time_causal(tmp_path):
    t = pd.to_datetime(["2019-11-09 10:00", "2019-11-09 10:00", "2019-11-09 10:30", "2019-11-09 12:00", "2019-11-10 09:00"])
    imps = pd.DataFrame({"imp_key": ["a", "a", "b", "c", "d"], "news_id": ["N1", "N2", "N1", "N1", "N2"], "user_id": ["u", "u", "u", "v", "u"],
                         "time": t, "label": [1, 0, 1, 0, 1]})
    imps.to_parquet(tmp_path / "impressions.parquet")
    pd.DataFrame({"news_id": ["N1", "N2"], "category": ["news", "sports"]}).to_parquet(tmp_path / "news.parquet")
    f = lv.extra_features(tmp_path).set_index(["imp_key", "news_id"])
    assert f.loc[("a", "N1"), "exposure_log"] == 0 and f.loc[("b", "N1"), "exposure_log"] == pytest.approx(np.log1p(1))
    prior = lv.LTR_DAY1_PRIOR
    assert f.loc[("b", "N1"), "ctr_1h"] == pytest.approx((1 + lv.CTR_A * prior) / (1 + lv.CTR_A))
    assert f.loc[("c", "N1"), "ctr_1h"] == pytest.approx(prior)          # nothing in the hour before 12:00
    assert f.loc[("b", "N1"), "user_log_cat_share"] == 1.0 and f.loc[("a", "N1"), "user_log_clicks"] == 0
    assert f.loc[("d", "N2"), "user_log_cat_share"] == 0.0               # u's earlier clicks were news, not sports
    assert f.loc[("c", "N1"), "age_hours"] == pytest.approx(2.0)
    imps.loc[4, "label"] = 0                                             # a later label must not change earlier rows
    imps.to_parquet(tmp_path / "impressions.parquet")
    g = lv.extra_features(tmp_path).set_index(["imp_key", "news_id"])
    assert g.drop(index=("d", "N2")).equals(f.drop(index=("d", "N2")))


def test_emotion_columns_are_user_relative_and_history_only():
    import inspect
    sig = inspect.signature(lv.emotion_columns)
    assert "label" not in sig.parameters and "truth" not in sig.parameters
    src = inspect.getsource(lv.emotion_columns)
    assert "go_pct_" not in src and "item_features" not in src      # no raw item-level column is copied in
