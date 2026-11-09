"""Tests for src/same_day.py: previous-days-only features never see an event of the row's own day."""
import numpy as np
import pandas as pd
import pytest

from src import same_day as sd


def frame():
    t = pd.to_datetime(["2019-11-09 10:00", "2019-11-09 22:00", "2019-11-10 01:00", "2019-11-10 08:00", "2019-11-10 09:00"])
    return pd.DataFrame({"news_id": ["N1", "N1", "N1", "N1", "N2"], "user_id": ["u", "v", "u", "w", "u"], "time": t, "label": [1, 0, 1, 1, 0]})


CATS = pd.Series({"N1": "news", "N2": "sports"})


def test_prev_day_values():
    f = sd.prev_day_features(frame(), CATS)
    prior = sd.LTR_DAY1_PRIOR
    assert f.loc[0, "pd_ctr_24h"] == pytest.approx(prior) and np.isnan(f.loc[0, "pd_age_hours"])   # first day: nothing earlier
    assert f.loc[3, "pd_ctr_24h"] == pytest.approx((1 + sd.CTR_A * prior) / (2 + sd.CTR_A))      # day 9: 1 click / 2 shows
    assert f.loc[3, "pd_exposure_log"] == pytest.approx(np.log1p(2))
    assert f.loc[3, "pd_ctr_6h"] == pytest.approx(sd.CTR_A * prior / (1 + sd.CTR_A)) and f.loc[3, "pd_shows_6h"] == pytest.approx(np.log1p(1))  # only 22:00 is within 6 h of 00:00
    assert list(f.columns) == sd.PD_COLS
    assert f.loc[3, "pd_age_hours"] == pytest.approx(14.0)                                        # first shown 10:00 the day before
    assert f.loc[4, "pd_user_log_clicks"] == pytest.approx(np.log1p(1)) and f.loc[4, "pd_user_log_cat_share"] == 0.0
    assert np.isnan(f.loc[4, "pd_age_hours"])                                                     # N2 never shown before 10 Nov


def test_same_day_events_never_change_a_row():
    base = sd.prev_day_features(frame(), CATS)
    g = frame()
    g.loc[2, "label"] = 0                      # change a same-day (10 Nov) label and add a same-day impression
    g = pd.concat([g, pd.DataFrame({"news_id": ["N1"], "user_id": ["u"], "time": pd.to_datetime(["2019-11-10 00:30"]), "label": [1]})],
                  ignore_index=True)
    after = sd.prev_day_features(g, CATS).iloc[:5]
    pd.testing.assert_frame_equal(base, after)


def test_catalogue_flags():
    assert sd.same_day_columns(["pop_ctr", "ctr_1h", "ctr_24h_z", "m2_score", "age_hours"]) == ["ctr_1h", "ctr_24h_z", "age_hours"]
