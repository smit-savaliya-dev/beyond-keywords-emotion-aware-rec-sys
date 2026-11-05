"""Subgroup evaluation for the week-7 experiments: metrics per group, the gain of one model over another per group with a user-level
bootstrap CI, trend tests and the shuffled-label control.

Inputs: saved prediction files (imp_key, news_id, score) or per-impression metric tables, and a group label per impression or candidate.
Outputs: one row per group with sizes, means, gains, CIs and a "too small" flag. Small groups are flagged, never dropped.
"""
import numpy as np
import pandas as pd
from scipy import stats as st

from .click_signal import permute_within_groups
from .stats import paired_test

MIN_GROUP = 500  # impressions; smaller groups are reported and flagged "too small"
METRICS = ("auc", "mrr", "ndcg@10")


def fast_metrics(keys, labels, scores, seed=42):
    """Per-impression AUC (ties count one half, as the harness), MRR and nDCG@10 (ties broken by a seeded random order).
    Rows of one impression must be consecutive. -> table imp_key, auc, mrr, ndcg@10 (NaN where undefined)."""
    keys, y, s = np.asarray(keys), np.asarray(labels, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    g = pd.factorize(keys)[0]
    n = np.bincount(g).astype(np.float64)
    pos = np.bincount(g, weights=y)
    avg_rank = pd.Series(s).groupby(g).rank(method="average").to_numpy()
    auc = np.full(len(n), np.nan)
    ok = (pos > 0) & (pos < n)
    rs = np.bincount(g, weights=avg_rank * y)
    auc[ok] = (rs[ok] - pos[ok] * (pos[ok] + 1) / 2) / (pos[ok] * (n[ok] - pos[ok]))
    tie = np.random.default_rng(seed).random(len(s))
    order = np.lexsort((tie, -s, g))                      # within impression: score descending, random tie-break
    rank = np.empty(len(s))
    rank[order] = np.arange(len(s)) - np.r_[0, np.cumsum(np.bincount(g))[:-1]][g[order]] + 1
    mrr = np.bincount(g, weights=y / rank) / np.where(pos > 0, pos, np.nan)
    dcg = np.bincount(g, weights=np.where(rank <= 10, y / np.log2(rank + 1), 0.0))
    ideal_k = np.minimum(pos, 10).astype(int)
    disc = np.r_[0.0, np.cumsum(1 / np.log2(np.arange(2, 12)))]
    ndcg = dcg / np.where(ideal_k > 0, disc[ideal_k], np.nan)
    first = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    return pd.DataFrame({"imp_key": keys[first], "auc": auc[g[first]], "mrr": mrr[g[first]], "ndcg@10": ndcg[g[first]]})


def restricted_metrics(truth, scores, mask, seed=42):
    """Per-impression metrics using only the candidate rows in `mask` (e.g. one exposure bucket). truth: harness order with label;
    scores: aligned score array."""
    t = truth.loc[np.asarray(mask), ["imp_key", "label"]]
    return fast_metrics(t["imp_key"].to_numpy(), t["label"].to_numpy(), np.asarray(scores)[np.asarray(mask)], seed)


def user_bootstrap_ci(values, users, n_boot=1000, seed=42):
    """95% CI of the impression-weighted mean of `values`, resampling users with replacement (NaN values dropped)."""
    v, u = np.asarray(values, dtype=np.float64), np.asarray(users)
    ok = np.isfinite(v)
    v, u = v[ok], u[ok]
    if len(v) == 0:
        return np.nan, np.nan
    codes, uniq = pd.factorize(u)
    sums, counts = np.bincount(codes, weights=v), np.bincount(codes).astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        means[b] = (w * sums).sum() / (w * counts).sum()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def group_gains(per_a, per_b, groups, users, metrics=METRICS, name_a="A", name_b="B", min_group=MIN_GROUP, n_boot=1000, seed=42):
    """B minus A per group. per_a, per_b: per-impression metric tables (imp_key + metrics); groups, users: Series indexed by imp_key.
    -> one row per group and metric: impressions, users, mean_a, mean_b, gain, bootstrap CI (users), paired-t p, dz, too_small."""
    m = per_a.merge(per_b, on="imp_key", suffixes=("_a", "_b"), validate="one_to_one")
    m["group"] = m["imp_key"].map(groups)
    m["user"] = m["imp_key"].map(users)
    rows = []
    for g, d in m.groupby("group", sort=False):
        for k in metrics:
            ok = d[f"{k}_a"].notna() & d[f"{k}_b"].notna()
            diff = (d[f"{k}_b"] - d[f"{k}_a"])[ok]
            lo, hi = user_bootstrap_ci(diff.to_numpy(), d.loc[ok, "user"].to_numpy(), n_boot, seed)
            t = paired_test(d.loc[ok, f"{k}_a"], d.loc[ok, f"{k}_b"])
            rows.append({"group": g, "metric": k, "model_a": name_a, "model_b": name_b, "impressions": int(ok.sum()),
                         "users": int(d.loc[ok, "user"].nunique()), "mean_a": t["mean_a"], "mean_b": t["mean_b"], "gain": t["mean_diff"],
                         "ci_low": lo, "ci_high": hi, "dz": t["dz"], "p": t["p"], "too_small": bool(ok.sum() < min_group)})
    return pd.DataFrame(rows)


def group_means(per, groups, users, metrics=METRICS, name="A", min_group=MIN_GROUP, n_boot=1000, seed=42):
    """Mean metric per group with a user-bootstrap CI. per: per-impression table; groups, users: Series indexed by imp_key."""
    d = per.assign(group=per["imp_key"].map(groups), user=per["imp_key"].map(users))
    rows = []
    for g, x in d.groupby("group", sort=False):
        for k in metrics:
            ok = x[k].notna()
            lo, hi = user_bootstrap_ci(x.loc[ok, k].to_numpy(), x.loc[ok, "user"].to_numpy(), n_boot, seed)
            rows.append({"group": g, "metric": k, "model": name, "impressions": int(ok.sum()), "users": int(x.loc[ok, "user"].nunique()),
                         "mean": float(x.loc[ok, k].mean()), "ci_low": lo, "ci_high": hi, "too_small": bool(ok.sum() < min_group)})
    return pd.DataFrame(rows)


def trend_tests(score, gain, groups=None, low="low", high="high"):
    """Does the gain rise with the score? Spearman rho (score vs gain, NaN pairs dropped) and Welch t-test of high minus low group."""
    s, g = np.asarray(score, dtype=np.float64), np.asarray(gain, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(g)
    rho = st.spearmanr(s[ok], g[ok])
    out = {"n": int(ok.sum()), "rho": float(rho.statistic), "rho_p": float(rho.pvalue)}
    if groups is not None:
        groups = np.asarray(groups)
        a, b = g[ok & (groups == low)], g[ok & (groups == high)]
        t = st.ttest_ind(b, a, equal_var=False)
        out.update({"high_minus_low": float(b.mean() - a.mean()), "hl_p": float(t.pvalue), "n_low": len(a), "n_high": len(b)})
    return out


def shuffled_labels(truth, seed=42):
    """Copy of the harness truth with click labels permuted within each impression: the null for every new score (plan v5)."""
    out = truth.copy()
    out["label"] = permute_within_groups(pd.factorize(truth["imp_key"])[0], truth["label"].to_numpy(), seed)
    return out


def paired_family(pairs, users, metric="auc", alpha=0.05):
    """One pre-declared family of paired comparisons. pairs: [(name, per_a, per_b)]; users: Series by imp_key.
    -> per comparison: per-impression paired t-test (B - A, CI, dz), user-level paired t-test, Holm across the family."""
    from .stats import holm
    rows = []
    for name, a, b in pairs:
        m = a[["imp_key", metric]].merge(b[["imp_key", metric]], on="imp_key", suffixes=("_a", "_b"), validate="one_to_one")
        t = paired_test(m[f"{metric}_a"], m[f"{metric}_b"])
        u = m.groupby(m["imp_key"].map(users))[[f"{metric}_a", f"{metric}_b"]].mean()
        ut = paired_test(u[f"{metric}_a"], u[f"{metric}_b"])
        rows.append({"comparison": name, "metric": metric, **{k: t[k] for k in ("n", "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high", "dz", "p")},
                     "users": ut["n"], "user_diff": ut["mean_diff"], "user_ci_low": ut["ci_low"], "user_ci_high": ut["ci_high"], "user_p": ut["p"]})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy())
    out["significant_holm"] = out["p_holm"] < alpha
    return out


def within_group_metrics(imp_keys, groups, labels, scores, seed=42):
    """Per (impression, group) metrics among the candidates of one group (e.g. one category) inside an impression.
    -> table imp_key, group, auc, mrr, ndcg@10."""
    key = np.asarray(imp_keys).astype(object) + "|" + np.asarray(groups).astype(str).astype(object)
    order = np.argsort(key, kind="stable")
    t = fast_metrics(key[order], np.asarray(labels)[order], np.asarray(scores)[order], seed)
    parts = t["imp_key"].str.rsplit("|", n=1)
    return t.assign(imp_key=parts.str[0], group=parts.str[1])[["imp_key", "group", "auc", "mrr", "ndcg@10"]]
