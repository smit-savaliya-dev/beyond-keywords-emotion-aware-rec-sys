"""Same-day signal audit (final check): which M4v6 features use clicks or impressions from the day being scored, and previous-days-only
versions of them.

A row on day D at time t. The M4v6 extra features count events strictly before t, so on day D they include earlier impressions and clicks
of day D itself (on the final test day: earlier dev-day labels). The "pd_" versions below use only events before 00:00 of day D.
Inputs: impressions (all parts). Output: frame imp_key, news_id + pd_ features.
"""
import numpy as np
import pandas as pd

from .config import LTR_DAY1_PRIOR, PROCESSED
from .ltr_v6 import CTR_A, WINDOWS_H, window_counts

# feature -> (uses same-day events?, what it uses). Covers every M4v6 / M5v6 input column.
CATALOGUE = {
    "pop_ctr": (False, "item CTR from days strictly before the row's day (M0, D64)"),
    "m1_score": (False, "TF-IDF similarity of history and candidate text"),
    "m2_score": (False, "Sentence-BERT similarity of history and candidate text"),
    "category": (False, "item metadata"),
    "subcategory": (False, "item metadata"),
    "user_cat_affinity": (False, "share of the user's history column in the candidate's category"),
    "history_len": (False, "length of the history column"),
    "title_len": (False, "item metadata"),
    "imp_size": (False, "size of this impression's own candidate list (no other events)"),
    "ctr_1h": (True, "item clicks / shows in the last 1 h before t, incl. same-day labels"),
    "shows_1h": (True, "item shows in the last 1 h before t (same-day impressions, no labels)"),
    "ctr_6h": (True, "item clicks / shows in the last 6 h before t, incl. same-day labels"),
    "shows_6h": (True, "item shows in the last 6 h before t (same-day impressions)"),
    "ctr_24h": (True, "item clicks / shows in the last 24 h before t, incl. same-day labels"),
    "shows_24h": (True, "item shows in the last 24 h before t (same-day impressions)"),
    "exposure_log": (True, "all item shows before t, incl. same day"),
    "age_hours": (True, "hours since the item's first impression; first impression can be earlier the same day"),
    "user_log_clicks": (True, "user's logged clicks before t, incl. same-day clicks"),
    "user_log_cat_share": (True, "share of the user's logged clicks before t in the candidate's category, incl. same day"),
    "ctr_24h_rank": (True, "within-impression rank of ctr_24h"),
    "ctr_24h_z": (True, "within-impression z-score of ctr_24h"),
    "pop_ctr_rank": (False, "within-impression rank of pop_ctr"),
    "pop_ctr_z": (False, "within-impression z-score of pop_ctr"),
    "m1_score_rank": (False, "within-impression rank"), "m1_score_z": (False, "within-impression z-score"),
    "m2_score_rank": (False, "within-impression rank"), "m2_score_z": (False, "within-impression z-score"),
    "user_cat_affinity_rank": (False, "within-impression rank"), "user_cat_affinity_z": (False, "within-impression z-score"),
}
EMOTION_NOTE = "M5v6 emotion columns (F2) use the history column and item text only: no same-day events"
PD_COLS = ["pd_ctr_1h", "pd_shows_1h", "pd_ctr_6h", "pd_shows_6h", "pd_ctr_24h", "pd_shows_24h", "pd_exposure_log", "pd_age_hours",
           "pd_user_log_clicks", "pd_user_log_cat_share"]
PD_NORM = ["pd_ctr_24h_rank", "pd_ctr_24h_z"]          # within-list versions of pd_ctr_24h (as ctr_24h_rank / _z)


def same_day_columns(features):
    return [c for c in features if CATALOGUE.get(c, (False,))[0]]


def prev_day_features(rows, categories):
    """Previous-days-only versions of the same-day M4v6 features (events before 00:00 of the row's day).
    rows: news_id, user_id, time, label (all parts, all candidates); categories: news_id -> category. -> frame of PD_COLS in row order."""
    t = pd.to_datetime(rows["time"])
    origin = t.min().normalize()
    sec = ((t - origin) / pd.Timedelta("1s")).to_numpy().astype(np.int64)
    start = ((t.dt.normalize() - origin) / pd.Timedelta("1s")).to_numpy().astype(np.int64)   # 00:00 of the row's day
    item = pd.factorize(rows["news_id"].astype(str))[0]
    user = pd.factorize(rows["user_id"].astype(str))[0]
    cat = pd.factorize(rows["news_id"].astype(str).map(categories).astype(str))[0]
    y = rows["label"].to_numpy(np.float64)
    out = {}
    for h in WINDOWS_H:                           # the last h hours before 00:00 of the row's day
        shows, clicks = window_counts(item, sec, y, item, start, h * 3600)
        out[f"pd_ctr_{h}h"] = ((clicks + CTR_A * LTR_DAY1_PRIOR) / (shows + CTR_A)).astype(np.float32)
        out[f"pd_shows_{h}h"] = np.log1p(shows).astype(np.float32)
    shows_all, _ = window_counts(item, sec, y, item, start, None)
    out["pd_exposure_log"] = np.log1p(shows_all).astype(np.float32)
    first = pd.Series(sec).groupby(item).transform("min").to_numpy()
    out["pd_age_hours"] = np.where(first < start, (start - first) / 3600.0, np.nan).astype(np.float32)   # NaN = not shown before today
    clicked = y == 1
    ucat = user * (cat.max() + 1) + cat
    n_cat, _ = window_counts(ucat[clicked], sec[clicked], y[clicked], ucat, start, None)
    n_all, _ = window_counts(user[clicked], sec[clicked], y[clicked], user, start, None)
    out["pd_user_log_clicks"] = np.log1p(n_all).astype(np.float32)
    out["pd_user_log_cat_share"] = np.where(n_all > 0, n_cat / np.maximum(n_all, 1), 0.0).astype(np.float32)
    return pd.DataFrame(out)


def all_prev_day_features(processed=PROCESSED):
    rows = pd.read_parquet(processed / "impressions.parquet", columns=["imp_key", "news_id", "user_id", "time", "label"])
    news = pd.read_parquet(processed / "news.parquet", columns=["news_id", "category"])
    out = prev_day_features(rows, news.set_index("news_id")["category"].astype(str))
    out.insert(0, "news_id", rows["news_id"].astype(str).to_numpy())
    out.insert(0, "imp_key", rows["imp_key"].astype(str).to_numpy())
    return out
