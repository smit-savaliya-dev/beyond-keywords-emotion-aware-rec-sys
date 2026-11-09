"""Week-8 tree models (plan v6): the extra no-emotion features of M4v6, the user-relative emotion features of M5v6, rolling-origin folds,
Optuna tuning with a fixed budget, fold evaluation and the final retrain on days 9-14.

Every feature is time-causal: it uses the history column, item text, or impressions with a timestamp strictly earlier than the row.
Inputs: impressions (all parts, for the time-causal counts), the M4 feature tables, emotion tables. Outputs: feature frames, fold scores.
"""
import json

import numpy as np
import pandas as pd

from .config import LTR_DAY1_PRIOR, PROCESSED

WINDOWS_H = [1, 6, 24]
CTR_A = 10.0      # smoothing as M0's frozen a; prior = LTR_DAY1_PRIOR, a fixed constant (never computed from data)
FOLDS = {"F1": "2019-11-11", "F2": "2019-11-12", "F3": "2019-11-13", "F4": "2019-11-14"}   # test day; train = earlier days
NORM_COLS = ["pop_ctr", "m1_score", "m2_score", "user_cat_affinity", "ctr_24h"]


def _seconds(t, origin):
    return ((pd.to_datetime(t) - origin) / pd.Timedelta("1s")).to_numpy().astype(np.int64)


def window_counts(event_keys, event_times, event_values, row_keys, row_times, width):
    """For each row: number and value-sum of events with the same key and time in [t - width, t) (strictly earlier).
    Keys are non-negative integers, times integer seconds below 10**7."""
    big = 10 ** 7
    ek = np.asarray(event_keys, dtype=np.int64) * big + np.asarray(event_times, dtype=np.int64)
    order = np.argsort(ek, kind="stable")
    ek, ev = ek[order], np.asarray(event_values, dtype=np.float64)[order]
    cum = np.r_[0.0, np.cumsum(ev)]
    rk = np.asarray(row_keys, dtype=np.int64) * big
    rt = np.asarray(row_times, dtype=np.int64)
    hi = np.searchsorted(ek, rk + rt, side="left")
    lo = np.searchsorted(ek, rk + (np.maximum(rt - width, 0) if width is not None else 0), side="left")
    return hi - lo, cum[hi] - cum[lo]


def extra_features(processed=PROCESSED):
    """Time-causal no-emotion features for every candidate row of every part. -> frame imp_key, news_id + features."""
    rows = pd.read_parquet(processed / "impressions.parquet", columns=["imp_key", "news_id", "user_id", "time", "label"])
    news = pd.read_parquet(processed / "news.parquet", columns=["news_id", "category"])
    origin = pd.to_datetime(rows["time"]).min().normalize()
    sec = _seconds(rows["time"], origin)
    item = pd.factorize(rows["news_id"].astype(str))[0]
    user = pd.factorize(rows["user_id"].astype(str))[0]
    cat = pd.factorize(rows["news_id"].astype(str).map(news.set_index("news_id")["category"].astype(str)))[0]
    y = rows["label"].to_numpy(np.float64)
    out = pd.DataFrame({"imp_key": rows["imp_key"].astype(str).to_numpy(), "news_id": rows["news_id"].astype(str).to_numpy()})
    for h in WINDOWS_H:
        shows, clicks = window_counts(item, sec, y, item, sec, h * 3600)
        out[f"ctr_{h}h"] = ((clicks + CTR_A * LTR_DAY1_PRIOR) / (shows + CTR_A)).astype(np.float32)
        out[f"shows_{h}h"] = np.log1p(shows).astype(np.float32)
    shows_all, _ = window_counts(item, sec, y, item, sec, None)
    out["exposure_log"] = np.log1p(shows_all).astype(np.float32)
    first = pd.Series(sec).groupby(item).transform("min").to_numpy()
    out["age_hours"] = ((sec - first) / 3600.0).astype(np.float32)      # first impression of the item is at or before t
    clicked = y == 1
    ucat = user * (cat.max() + 1) + cat
    n_cat, _ = window_counts(ucat[clicked], sec[clicked], y[clicked], ucat, sec, None)
    n_all, _ = window_counts(user[clicked], sec[clicked], y[clicked], user, sec, None)
    out["user_log_clicks"] = np.log1p(n_all).astype(np.float32)
    out["user_log_cat_share"] = np.where(n_all > 0, n_cat / np.maximum(n_all, 1), 0.0).astype(np.float32)
    return out


def within_impression(frame, cols=NORM_COLS):
    """Rank (0-1) and z-score of each column inside its candidate list."""
    g = frame.groupby("imp_key", sort=False)
    out = {}
    for c in cols:
        x = frame[c].astype(np.float64)
        out[f"{c}_rank"] = (g[c].rank(pct=True)).astype(np.float32)
        mu, sd = g[c].transform("mean"), g[c].transform("std").fillna(0)
        out[f"{c}_z"] = np.where(sd > 0, (x - mu) / sd.where(sd > 0, 1.0), 0.0).astype(np.float32)
    return pd.DataFrame(out, index=frame.index)


def fold_masks(days, fold):
    """(train mask, test mask) of a rolling-origin fold over fit + valid rows."""
    test = pd.Timestamp(FOLDS[fold])
    d = pd.DatetimeIndex(pd.to_datetime(np.asarray(days))).normalize()
    return np.asarray(d < test), np.asarray(d == test)


def search_space(trial, strong=False):
    """The pre-declared Optuna space (plan v6); `strong` = the F4 variant's stronger regularisation."""
    return {"num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 300 if strong else 20, 1000 if strong else 500, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4 if strong else 0.5, 0.8 if strong else 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0 if strong else 1e-3, 20.0 if strong else 10.0, log=True)}


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def persona_model(fit_ctx, min_items=5, k=None, seed=42, n_init=10, sample=20000):
    """The E8 personas refit on fit users (history only): -> (k-means model, fit mean, fit SD). k defaults to E8's choice."""
    from .behaviour import fit_personas, window_profiles
    prof, used = window_profiles(fit_ctx.hist_rows, fit_ctx.vectors, 50, 0.9)
    first = pd.Series(np.arange(len(fit_ctx.prof_users))).groupby(fit_ctx.prof_users).first().to_numpy()
    rows = first[used[first] >= min_items]
    x = prof[rows]
    mu, sd = x.mean(axis=0), x.std(axis=0)
    best, _, models = fit_personas((x - mu) / sd, [k] if k else [3, 4, 5, 6], seed, n_init, sample)
    return models[best], mu, sd


def emotion_columns(ctx, km, mu, sd, resid, raw28, valence, seed=42, min_items=5):
    """User-relative emotion features of M5v6 for one part, in harness order (no raw item-level column; plan v6 F1-F4).
    raw28: GoEmotions raw scores per news row; valence: go_vad_valence per news row; resid: residual go_pct vectors (E5)."""
    from .behaviour import history_consistency, window_profiles
    from .congruence import congruence_features, cosine_rows, user_profiles
    from .config import LTR_CONGRUENCE_MEASURES
    from .emotion import collapse
    g, it = ctx.groups, ctx.item_idx
    vec = {"pct": ctx.vectors, "six": collapse(raw28), "valence": valence}
    cong, _ = congruence_features(g, it, ctx.hist_rows, vec, LTR_CONGRUENCE_MEASURES, shuffle_seed=seed)
    out = {}
    for base in ("cong_pct_cos", "cong_six_negkl", "cong_valence_absdist"):
        for suffix in ("", "_personal", "_userspecific"):
            out[f"{base}{suffix}"] = cong[f"{base}{suffix}"].astype(np.float32)
    pair, _ = history_consistency(ctx.hist_rows, ctx.vectors)
    out["consistency"] = pair[g].astype(np.float32)
    out["consistency_x_personal"] = (pair[g] * out["cong_pct_cos_personal"]).astype(np.float32)
    prof, used = window_profiles(ctx.hist_rows, ctx.vectors, 50, 0.9)
    persona = np.where(used >= min_items, km.predict((prof - mu) / sd), -1)
    out["persona"] = persona[g].astype(np.int16)
    rprof, _ = user_profiles(ctx.hist_rows, resid, np.zeros(resid.shape[1]), decay=0.9, max_items=50)
    out["resid_cong"] = cosine_rows(resid[it], rprof[g]).astype(np.float32)
    p28, _ = user_profiles(ctx.hist_rows, raw28, raw28.mean(axis=0), decay=0.9, max_items=50)
    for j in range(raw28.shape[1]):
        out[f"prod_{j:02d}"] = (p28[g, j] * raw28[it, j]).astype(np.float32)
    return out


def train_fixed(train, features, params, seed, rounds, categorical=("category", "subcategory")):
    """One LambdaRank model with a fixed number of rounds (the final retrain on days 9-14, no early stopping)."""
    import lightgbm as lgb
    from .config import LTR_NEG_PER_CLICK, LTR_THREADS
    from .lightgbm_ranker import group_sizes, sample_negatives
    keep = sample_negatives(train["imp_key"].to_numpy(), train["label"].to_numpy(), LTR_NEG_PER_CLICK, seed)
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=int(rounds), random_state=seed, n_jobs=LTR_THREADS, deterministic=True,
                           force_row_wise=True, verbose=-1, **params)
    model.fit(train.loc[keep, features], train["label"].to_numpy()[keep], group=group_sizes(train["imp_key"].to_numpy()[keep]),
              categorical_feature=[c for c in features if c in categorical])
    return model.booster_
