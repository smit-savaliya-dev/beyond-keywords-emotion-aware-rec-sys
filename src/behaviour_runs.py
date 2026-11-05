"""Shared run context for the week-7 experiments: one part's candidates, profiles, emotion vectors and M2 scores, plus the M3-style
fusion and fast evaluation used by every experiment script.

Inputs: data/processed tables and the saved M2 scores (valid, dev) or the M4 feature table (fit, which holds the same M2 score).
Outputs: a Context object; fused scores; per-impression metric tables (AUC identical to the harness).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baselines import load_part
from .behaviour import blend_profiles, personal_match, same_day_profiles, window_profiles
from .config import GO_PCT_COLUMNS, LTR_FEATURE_FILE, M2_NAME, PROCESSED, RESULTS
from .congruence import history_index_lists
from .evaluate import attach_scores
from .fusion import fuse, minmax_within
from .subgroups import fast_metrics


@dataclass
class Context:
    part: str
    truth: pd.DataFrame          # harness order: imp_key, news_id, label, position
    imp_codes: np.ndarray        # impression number per candidate row (0..n_imp-1, in truth order)
    groups: np.ndarray           # profile row per candidate row
    item_idx: np.ndarray         # row in `vectors` per candidate row
    hist_rows: list              # per profile row: history item rows, oldest -> newest
    prof_users: np.ndarray       # user per profile row
    imps: pd.DataFrame           # per impression (imp_codes order): imp_key, user_id, time, prof_row
    vectors: np.ndarray          # emotion vectors (8 go_pct), rows = news_ids
    news_ids: pd.Index
    m2: np.ndarray | None = None # M2 score per candidate row

    @property
    def users_by_imp(self):
        return pd.Series(self.imps["user_id"].to_numpy(), index=self.imps["imp_key"].to_numpy())


def load_context(part, processed=PROCESSED, with_m2=True):
    """Everything a behaviour experiment needs for one part. Only that part's behaviours and impressions are read."""
    feats = pd.read_parquet(processed / "news_emotion_features_v2.parquet", columns=["news_id", *GO_PCT_COLUMNS])
    index = pd.Index(feats["news_id"])
    vectors = feats[GO_PCT_COLUMNS].to_numpy(np.float64)
    truth, groups, histories, users = load_part(part, processed)
    item_idx = index.get_indexer(truth["news_id"])
    if (item_idx < 0).any():
        raise ValueError("candidates without emotion vectors")
    imp_codes, imp_keys = pd.factorize(truth["imp_key"])
    first = np.flatnonzero(np.r_[True, imp_codes[1:] != imp_codes[:-1]])
    beh = pd.read_parquet(processed / "behaviors.parquet", columns=["imp_key", "part", "user_id", "time"], filters=[("part", "==", part)])
    if set(beh["part"].astype(str)) != {part}:
        raise ValueError("behaviors of other parts loaded")
    beh = beh.set_index(beh["imp_key"].astype(str))
    imps = pd.DataFrame({"imp_key": np.asarray(imp_keys), "prof_row": groups[first]})
    imps["user_id"] = beh["user_id"].astype(str).reindex(imps["imp_key"]).to_numpy()
    imps["time"] = pd.to_datetime(beh["time"].reindex(imps["imp_key"]).to_numpy())
    ctx = Context(part, truth, imp_codes, groups, item_idx, history_index_lists(histories, index), users, imps, vectors, index)
    if with_m2:
        ctx.m2 = m2_scores(ctx, processed)
    return ctx


def m2_scores(ctx, processed=PROCESSED):
    """M2 per candidate: the saved valid/dev scores, or on fit the m2_score column of the M4 feature table (same computation, D67b)."""
    if ctx.part == "fit":
        f = pd.read_parquet(processed / LTR_FEATURE_FILE.format(part="fit"), columns=["imp_key", "news_id", "m2_score"])
        if f["imp_key"].tolist() != ctx.truth["imp_key"].tolist():
            raise ValueError("fit feature rows are not in harness order")
        return f["m2_score"].to_numpy(np.float64)
    return attach_scores(ctx.truth, pd.read_parquet(RESULTS / "scores" / f"{M2_NAME}__{ctx.part}.parquet"))["score"].to_numpy()


def emotion_score(ctx, size=50, decay=0.9, same_day=None, a=1.0):
    """The M3 personal match with a chosen history window, optionally blended with a same-day profile (per impression)."""
    prof, _ = window_profiles(ctx.hist_rows, ctx.vectors, size, decay)
    if same_day is None or a == 1.0:
        return personal_match(ctx.groups, ctx.item_idx, prof, ctx.vectors)
    per_imp = blend_profiles(prof[ctx.imps["prof_row"].to_numpy()], same_day, a)
    return personal_match(ctx.imp_codes, ctx.item_idx, per_imp, ctx.vectors)


def same_day(ctx, processed=PROCESSED):
    """Same-day profile per impression from the part's own clicks in strictly earlier impressions (time-causal, plan v5 E2).
    -> (profiles, clicks used, latest click time)."""
    clicks = ctx.truth.loc[ctx.truth["label"].to_numpy() == 1, ["imp_key"]].copy()
    clicks["item"] = ctx.item_idx[ctx.truth["label"].to_numpy() == 1]
    meta = ctx.imps.set_index("imp_key")
    clicks["user_id"] = meta["user_id"].reindex(clicks["imp_key"]).to_numpy()
    clicks["time"] = meta["time"].reindex(clicks["imp_key"]).to_numpy()
    return same_day_profiles(clicks, ctx.imps[["user_id", "time"]], ctx.vectors)


def fused_scores(ctx, emotion, alpha, alpha_by_imp=None):
    """alpha * minmax(M2) + (1 - alpha) * minmax(emotion) within each impression; alpha_by_imp gives one alpha per impression."""
    base = minmax_within(ctx.imp_codes, ctx.m2)
    emo = minmax_within(ctx.imp_codes, emotion)
    if alpha_by_imp is not None:
        a = np.asarray(alpha_by_imp, dtype=np.float64)[ctx.imp_codes]
        return a * base + (1.0 - a) * emo
    return fuse(base, emo, alpha)


def metrics(ctx, scores, labels=None, seed=42):
    """Per-impression AUC, MRR, nDCG@10 of a score vector (labels default to the true clicks)."""
    y = ctx.truth["label"].to_numpy() if labels is None else labels
    return fast_metrics(ctx.truth["imp_key"].to_numpy(), y, scores, seed)


def exposure_table(processed=PROCESSED):
    """Exposure of every candidate row of every part (strictly earlier impressions showing the item) and Q, the median fit exposure
    above 0. -> (Series by (imp_key, news_id), Q)."""
    from .behaviour import exposure_counts
    rows = pd.read_parquet(processed / "impressions.parquet", columns=["imp_key", "part", "news_id", "time"])
    rows["exposure"] = exposure_counts(rows["news_id"].astype(str).to_numpy(), rows["time"].to_numpy())
    q = float(np.median(rows.loc[(rows["part"].astype(str) == "fit") & (rows["exposure"] > 0), "exposure"]))
    return rows.set_index([rows["imp_key"].astype(str), rows["news_id"].astype(str)])["exposure"], q
