"""Tests for src/behaviour.py (consistency, windows, same-day profiles, drift, exposure, personas) and its leakage rules, with small fakes:
no internet, GPU or real lexicons."""
import numpy as np
import pandas as pd
import pytest

from src import behaviour as bh

V = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])


def test_consistency_closed_forms_match_brute_force():
    hist = [np.array([0, 1, 2, 3]), np.array([3]), np.array([-1, 0, 3]), np.array([], dtype=np.int64)]
    pair, cent = bh.history_consistency(hist, V)
    u = V / np.linalg.norm(V, axis=1, keepdims=True)
    idx = [0, 1, 2, 3]
    brute = np.mean([u[i] @ u[j] for i in idx for j in idx if i < j])
    c = u[idx].mean(axis=0)
    assert pair[0] == pytest.approx(brute) and cent[0] == pytest.approx(np.mean(u[idx] @ c / np.linalg.norm(c)))
    assert np.isnan(pair[1]) and np.isnan(pair[3]) and pair[2] == pytest.approx(1.0)   # unknown items are skipped


def test_tertiles_use_fixed_cuts_and_flag_missing():
    cuts = bh.tertile_cuts(np.arange(9.0))
    assert cuts == pytest.approx((8 / 3, 16 / 3))
    assert list(bh.assign_tertile([0, 4, 8, np.nan], cuts)) == ["low", "mid", "high", "no_history"]


def test_personal_match_is_raw_minus_generic():
    prof = np.array([[1.0, 0.0]])
    out = bh.personal_match(np.array([0, 0]), np.array([0, 1]), prof, V)
    mean = V.mean(axis=0)
    gen0 = V[0] @ mean / (np.linalg.norm(V[0]) * np.linalg.norm(mean))
    assert out[0] == pytest.approx(1.0 - gen0)


def test_window_profile_keeps_newest_items_only():
    prof, used = bh.window_profiles([np.array([0, 1, 3])], V, size=1, decay=1.0)
    assert used[0] == 1 and np.allclose(prof[0], V[3])


def make_log():
    t = pd.Timestamp("2019-11-15 10:00")
    clicks = pd.DataFrame({"user_id": ["u", "u", "u", "v"], "time": [t, t + pd.Timedelta("1h"), t + pd.Timedelta("2h"), t],
                           "item": [0, 1, 3, 2]})
    imps = pd.DataFrame({"user_id": ["u", "u", "u", "u", "v"],
                         "time": [t, t + pd.Timedelta("90min"), t + pd.Timedelta("2h"), t + pd.Timedelta("1D"), t + pd.Timedelta("1min")]})
    return clicks, imps


def test_same_day_profile_uses_only_strictly_earlier_clicks_of_that_day():
    clicks, imps = make_log()
    prof, used, last = bh.same_day_profiles(clicks, imps, V, decay=0.5)
    assert np.isnan(prof[0]).all() and used[0] == 0                 # its own click has the same time: excluded
    assert used[1] == 2 and np.allclose(prof[1], (0.5 * V[0] + V[1]) / 1.5)
    assert used[2] == 2                                               # the 12:00 click is not strictly earlier than 12:00
    assert np.isnan(prof[3]).all()                                    # next day: no same-day clicks
    assert used[4] == 1 and np.allclose(prof[4], V[2])
    ok = used > 0
    assert (last[ok] < pd.to_datetime(imps["time"]).to_numpy()[ok]).all()


def test_same_day_profile_ignores_later_clicks():
    clicks, imps = make_log()
    before, _, _ = bh.same_day_profiles(clicks, imps, V)
    later = pd.DataFrame({"user_id": ["u"], "time": [pd.Timestamp("2019-11-15 23:00")], "item": [2]})
    after, _, _ = bh.same_day_profiles(pd.concat([clicks, later], ignore_index=True), imps, V)
    assert np.allclose(np.nan_to_num(before), np.nan_to_num(after))


def test_blend_only_where_a_same_day_profile_exists():
    long = np.ones((2, 2))
    same = np.array([[0.0, 0.0], [np.nan, np.nan]])
    out = bh.blend_profiles(long, same, 0.25)
    assert np.allclose(out[0], 0.25) and np.allclose(out[1], 1.0)


def test_drift_needs_ten_items_and_compares_recent_with_older():
    vec = np.array([[1.0, 0.0], [0.0, 1.0]])
    stable = np.zeros(10, dtype=np.int64)
    shifted = np.r_[np.zeros(5, dtype=np.int64), np.ones(5, dtype=np.int64)]
    d = bh.drift_scores([stable, shifted, np.zeros(9, dtype=np.int64)], vec)
    assert d[0] == pytest.approx(0.0) and d[1] == pytest.approx(1.0) and np.isnan(d[2])


def test_exposure_counts_strictly_earlier_impressions_only():
    t = pd.to_datetime(["2019-11-15 10:00", "2019-11-15 10:00", "2019-11-15 11:00", "2019-11-15 09:00", "2019-11-15 12:00"])
    c = bh.exposure_counts(["a", "a", "a", "b", "b"], t)
    assert list(c) == [0, 0, 2, 0, 1]
    assert list(bh.exposure_bucket([0, 1, 3, 4], 3)) == ["new", "few", "few", "many"]


def test_history_features_do_not_read_labels():
    import inspect
    for fn in (bh.history_consistency, bh.drift_scores, bh.window_profiles):
        assert "label" not in inspect.signature(fn).parameters


def test_time_slots_cover_every_hour():
    slots = {"night": [0, 5], "morning": [6, 11], "afternoon": [12, 17], "evening": [18, 23]}
    t = pd.date_range("2019-11-15", periods=24, freq="h")
    out = bh.time_slot(t, slots)
    assert out[3] == "night" and out[6] == "morning" and out[12] == "afternoon" and out[23] == "evening" and None not in out


def test_personas_pick_the_clear_k_and_match_clusters():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(c, 0.05, size=(60, 2)) for c in ((0, 0), (3, 0), (0, 3))])
    k, scores, models = bh.fit_personas(x, [2, 3, 4], seed=0, n_init=3, sample=1000)
    assert k == 3 and scores[3] > scores[2]
    c = models[3].cluster_centers_
    b = c[::-1]
    assert np.allclose(b[bh.match_clusters(c, b)], c)
    assert bh.persona_names([[2.0, -1.0, 1.0]], ["fear", "joy", "anger"]) == ["fear + anger"]


def test_residual_vectors_remove_the_category_mean():
    vec = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [5.0, 5.0]])
    out = bh.residual_vectors(vec, np.array(["a", "a", "b", "c"]), shown=[0, 1, 2])
    assert np.allclose(out[:2], [[-1, 0], [1, 0]]) and np.allclose(out[2], 0)
    assert np.allclose(out[3], vec[3] - vec[:3].mean(axis=0))      # category c never shown in fit


def test_top_k_distinct_counts_labels_in_the_top_only():
    codes = np.array([0, 0, 0, 1, 1])
    scores = np.array([3.0, 2.0, 1.0, 1.0, 2.0])
    labels = np.array(["joy", "joy", "fear", "fear", "anger"])
    assert list(bh.top_k_distinct(codes, scores, labels, k=2)) == [1, 2]
    assert list(bh.top_k_distinct(codes, scores, labels, k=3)) == [2, 2]
