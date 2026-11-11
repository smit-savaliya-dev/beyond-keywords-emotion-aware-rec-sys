"""Final multi-metric evaluation from SAVED final dev predictions (no training, no new dev scoring of any model).

Ranking metrics come from src.metrics.impression_metrics through the same loop, sort order and seed as src.evaluate.evaluate, so AUC,
MRR, nDCG, P, R and HR equal the main tables exactly. The top-10 list used for diversity and coverage is taken in that same tie-break
order (a second generator with the same seed, consumed in lock-step).
"""
import numpy as np
import pandas as pd

from .config import K_VALUES, PROCESSED, SEED
from .evaluate import attach_scores
from .metrics import impression_metrics, ranked_labels
from .stats import holm, paired_test

TOP = 10
RANKING = ["auc", "mrr", "ndcg@5", "ndcg@10", "p@5", "r@10", "hr@5", "hr@10"]
LIST = ["cat_div@10", "emo_div@10"]
PER_IMP = RANKING + LIST
METRIC_NAMES = {"auc": "AUC", "mrr": "MRR", "ndcg@5": "nDCG@5", "ndcg@10": "nDCG@10", "p@5": "Precision@5", "r@10": "Recall@10",
                "hr@5": "Hit rate@5", "hr@10": "Hit rate@10", "cat_div@10": "Category diversity@10", "emo_div@10": "Emotion diversity@10",
                "coverage@10": "Catalogue coverage@10"}


def evaluate_with_lists(truth, scores, categories, emotions, ks=K_VALUES, seed=SEED, top=TOP):
    """Same as src.evaluate.evaluate (identical metrics and tie-breaks) plus, per impression, the number of distinct categories and
    emotion labels in the top `top`, and the set of articles that reach any top `top`. -> (per-impression frame, set of news ids)."""
    df = attach_scores(truth, scores).sort_values(["imp_key", "position"], kind="mergesort")
    keys = df["imp_key"].to_numpy()
    y = df["label"].to_numpy(dtype=float)
    s = df["score"].to_numpy(dtype=float)
    ids = df["news_id"].to_numpy()
    cat = pd.factorize(df["news_id"].map(categories).astype(str))[0]
    emo = pd.factorize(df["news_id"].map(emotions).astype(str))[0]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    rng, rng_order = np.random.default_rng(seed), np.random.default_rng(seed)    # lock-step: same draws, same order
    rows, shown = [], set()
    for a, b in zip(starts, ends):
        m = impression_metrics(y[a:b], s[a:b], ks, rng)
        order = ranked_labels(np.arange(b - a), s[a:b], rng_order).astype(int)[:top] + a
        m["cat_div@10"] = float(len(np.unique(cat[order])))
        m["emo_div@10"] = float(len(np.unique(emo[order])))
        shown.update(ids[order])
        rows.append(m)
    per = pd.DataFrame(rows)
    per.insert(0, "imp_key", keys[starts])
    return per, shown


def twin_gains(pairs, per, users, coverage, alpha=0.05):
    """pairs: [(pair name, twin code, emotion code)]; per: code -> per-impression frame; users: imp_key -> user_id; coverage: code -> float.
    -> one row per pair x metric: gain (emotion - twin), impression and user-level 95% CI, paired t-test p, dz; Holm over all
    pair x per-impression-metric tests. Coverage is one number per model: gain only, not tested."""
    rows = []
    for name, a, b in pairs:
        m = per[a][["imp_key"] + PER_IMP].merge(per[b][["imp_key"] + PER_IMP], on="imp_key", suffixes=("_a", "_b"), validate="one_to_one")
        g = m["imp_key"].map(users)
        for k in PER_IMP:
            t = paired_test(m[f"{k}_a"], m[f"{k}_b"])
            u = m.groupby(g)[[f"{k}_a", f"{k}_b"]].mean()
            ut = paired_test(u[f"{k}_a"], u[f"{k}_b"])
            rows.append({"pair": name, "twin": a, "emotion_model": b, "metric": k, "metric_name": METRIC_NAMES[k], "n": t["n"],
                         "twin_value": t["mean_a"], "emotion_value": t["mean_b"], "gain": t["mean_diff"], "ci_low": t["ci_low"],
                         "ci_high": t["ci_high"], "users": ut["n"], "user_gain": ut["mean_diff"], "user_ci_low": ut["ci_low"],
                         "user_ci_high": ut["ci_high"], "p": t["p"], "dz": t["dz"], "tested": True})
        rows.append({"pair": name, "twin": a, "emotion_model": b, "metric": "coverage@10", "metric_name": METRIC_NAMES["coverage@10"],
                     "twin_value": coverage[a], "emotion_value": coverage[b], "gain": coverage[b] - coverage[a], "tested": False})
    out = pd.DataFrame(rows)
    t = out["tested"].astype(bool)
    out.loc[t, "p_holm"] = holm(out.loc[t, "p"].to_numpy())
    out["significant_holm"] = t & (out["p_holm"] < alpha)
    return out


def load_users(part, processed=PROCESSED):
    imps = pd.read_parquet(processed / "impressions.parquet", columns=["imp_key", "user_id", "part"], filters=[("part", "==", part)])
    imps = imps[imps["part"].astype(str) == part].drop_duplicates("imp_key")
    return pd.Series(imps["user_id"].astype(str).to_numpy(), index=imps["imp_key"].astype(str).to_numpy())
