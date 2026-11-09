"""User-behaviour features for the week-7 experiments (plan v5, spec v5): consistency, window and same-day profiles, drift, item exposure,
time slots and personas. Every feature is time-causal: it uses the history column (which ends before the log period) or events with a
timestamp strictly earlier than the impression.

Inputs: history row lists, item emotion vectors, click and impression logs. Outputs: one value per profile row, impression or candidate.
Profiles reuse src.congruence, so there is one profile definition in the project.
"""
import numpy as np
import pandas as pd

from .congruence import cosine_rows, user_profiles


def unit_rows(m):
    """Rows scaled to length 1; a zero row stays zero."""
    m = np.asarray(m, dtype=np.float64)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n > 0, n, 1.0)


def history_consistency(hist_rows, vectors, max_items=50, min_items=2):
    """Per profile row: (mean pairwise cosine, mean cosine to the centroid) of the last `max_items` known history items.
    With unit vectors u and s = sum(u): pairwise = (|s|^2 - n) / (n(n - 1)) and centroid = |s| / n. NaN below `min_items`."""
    u = unit_rows(vectors)
    pair, cent = np.full(len(hist_rows), np.nan), np.full(len(hist_rows), np.nan)
    for g, h in enumerate(hist_rows):
        h = np.asarray(h, dtype=np.int64)
        h = h[h >= 0][-max_items:]
        n = len(h)
        if n >= min_items:
            s = u[h].sum(axis=0)
            ss = float(s @ s)
            pair[g] = (ss - n) / (n * (n - 1))
            cent[g] = np.sqrt(ss) / n
    return pair, cent


def tertile_cuts(values):
    """The 1/3 and 2/3 quantiles of the finite values (computed on fit only)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, 1 / 3)), float(np.quantile(v, 2 / 3))


def assign_tertile(values, cuts, missing="no_history"):
    """'low', 'mid' or 'high' by the fixed cut-offs; NaN -> `missing`."""
    v = np.asarray(values, dtype=np.float64)
    out = np.where(v <= cuts[0], "low", np.where(v <= cuts[1], "mid", "high")).astype(object)
    out[~np.isfinite(v)] = missing
    return out


def window_profiles(hist_rows, vectors, size, decay):
    """Emotion profile of the newest `size` known history items, weight decay**k (k = 0 newest); cold start = corpus mean."""
    vectors = np.asarray(vectors, dtype=np.float64)
    return user_profiles(hist_rows, vectors, vectors.mean(axis=0), decay=decay, max_items=size)


def personal_match(groups, item_idx, profiles, vectors, chunk=500_000):
    """The M3 score per candidate: cos(item, profile) - cos(item, corpus mean)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    mean = vectors.mean(axis=0)
    groups, item_idx = np.asarray(groups), np.asarray(item_idx)
    out = np.empty(len(groups), dtype=np.float64)
    for s in range(0, len(groups), chunk):
        e = min(s + chunk, len(groups))
        item = vectors[item_idx[s:e]]
        out[s:e] = cosine_rows(item, profiles[groups[s:e]]) - cosine_rows(item, np.broadcast_to(mean, item.shape))
    return out


def same_day_profiles(clicks, imps, vectors, decay=0.9):
    """Profile of the user's clicks in STRICTLY earlier impressions of the same calendar day, newest weight 1, weight decay**k.
    clicks: user_id, time, item (row in vectors), in any order. imps: user_id, time (one row per impression, any order).
    -> (profiles in imps order with NaN rows where no earlier click exists, clicks used, time of the latest click used or NaT)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    c = pd.DataFrame({"user": clicks["user_id"].astype(str).to_numpy(), "time": pd.to_datetime(clicks["time"]).to_numpy(),
                      "item": np.asarray(clicks["item"], dtype=np.int64)})
    c["key"] = c["user"] + "|" + c["time"].dt.normalize().astype(str)
    c = c.sort_values(["key", "time"], kind="mergesort").reset_index(drop=True)
    k = c.groupby("key").cumcount().to_numpy(np.float64)
    # decayed sum up to click j: sum_i v_i * decay**(j - i) = decay**j * sum_i v_i * decay**(-i); the decay**j cancels in the mean
    w = decay ** (-k)
    acc = pd.DataFrame(vectors[c["item"].to_numpy()] * w[:, None]).groupby(c["key"].to_numpy()).cumsum().to_numpy()
    wsum = pd.Series(w).groupby(c["key"].to_numpy()).cumsum().to_numpy()
    i = pd.DataFrame({"user": imps["user_id"].astype(str).to_numpy(), "time": pd.to_datetime(imps["time"]).to_numpy(),
                      "row": np.arange(len(imps))})
    i["key"] = i["user"] + "|" + i["time"].dt.normalize().astype(str)
    c["j"] = np.arange(len(c))
    m = pd.merge_asof(i.sort_values("time", kind="mergesort"), c[["key", "time", "j"]].sort_values("time", kind="mergesort"),
                      on="time", by="key", allow_exact_matches=False, direction="backward")    # strictly earlier clicks only
    m = m.sort_values("row")
    j = m["j"].to_numpy()
    ok = np.isfinite(j)
    jj = j[ok].astype(np.int64)
    out = np.full((len(i), vectors.shape[1]), np.nan)
    out[ok] = acc[jj] / wsum[jj, None]
    used = np.zeros(len(i), dtype=np.int64)
    used[ok] = k[jj].astype(np.int64) + 1
    last = np.full(len(i), np.datetime64("NaT"), dtype="datetime64[ns]")
    last[ok] = c["time"].to_numpy()[jj]
    return out, used, last


def blend_profiles(long_term, same_day, a):
    """a * long-term + (1 - a) * same-day where a same-day profile exists; long-term elsewhere."""
    out = np.array(long_term, dtype=np.float64, copy=True)
    has = np.isfinite(same_day).all(axis=1)
    out[has] = a * long_term[has] + (1.0 - a) * same_day[has]
    return out


def drift_scores(hist_rows, vectors, recent=5, older_max=45, min_items=10):
    """Per profile row: mean pairwise cosine within the last `recent` known items minus cosine(mean recent, mean older).
    NaN with fewer than `min_items` known items."""
    u = unit_rows(vectors)
    v = np.asarray(vectors, dtype=np.float64)
    out = np.full(len(hist_rows), np.nan)
    for g, h in enumerate(hist_rows):
        h = np.asarray(h, dtype=np.int64)
        h = h[h >= 0]
        if len(h) < min_items:
            continue
        r, o = h[-recent:], h[:-recent][-older_max:]
        s = u[r].sum(axis=0)
        within = (float(s @ s) - len(r)) / (len(r) * (len(r) - 1))
        a, b = v[r].mean(axis=0), v[o].mean(axis=0)
        out[g] = within - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return out


def exposure_counts(news_ids, times):
    """Per candidate row: number of impressions with a STRICTLY earlier timestamp that showed the same item (all rows given count)."""
    t = pd.DataFrame({"n": np.asarray(news_ids).astype(str), "t": pd.to_datetime(times).to_numpy()})
    return (t.groupby("n")["t"].rank(method="min").to_numpy() - 1).astype(np.int64)


def exposure_bucket(counts, q):
    """'new' (0), 'few' (1 to q) or 'many' (> q)."""
    c = np.asarray(counts)
    return np.where(c == 0, "new", np.where(c <= q, "few", "many")).astype(object)


def time_slot(times, slots):
    """Name of the hour slot of each timestamp; slots = {name: [first hour, last hour]}."""
    hours = np.asarray(pd.DatetimeIndex(pd.to_datetime(times)).hour)
    out = np.empty(len(hours), dtype=object)
    for name, (lo, hi) in slots.items():
        out[(hours >= lo) & (hours <= hi)] = name
    return out


def fit_personas(x, ks, seed, n_init, sample):
    """k-means for each k, silhouette on a seeded sample. -> (best k, {k: silhouette}, {k: fitted model})."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(sample, len(x)), replace=False)
    scores, models = {}, {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(x)
        models[k] = km
        scores[k] = float(silhouette_score(x[idx], km.labels_[idx]))
    best = max(ks, key=lambda k: (scores[k], -k))
    return best, scores, models


def persona_names(centroids_z, emotions, top=2):
    """Plain names from the emotions with the highest centroid z-scores, e.g. 'fear + nervousness'."""
    return [" + ".join(np.asarray(emotions)[np.argsort(-c)[:top]]) for c in np.asarray(centroids_z)]


def match_clusters(a, b):
    """Order of the rows of b that best matches the rows of a (Hungarian on Euclidean distance)."""
    from scipy.optimize import linear_sum_assignment
    cost = np.linalg.norm(np.asarray(a)[:, None, :] - np.asarray(b)[None, :, :], axis=2)
    _, cols = linear_sum_assignment(cost)
    return cols


def residual_vectors(vectors, cats, shown):
    """Item vector minus the mean vector of its category over the items in `shown` (items of fit impressions); an unseen category is centred
    on the mean of the shown items. Used to ask whether emotion adds anything beyond topic (plan v5 E5)."""
    vectors, cats, shown = np.asarray(vectors, dtype=np.float64), np.asarray(cats), np.unique(shown)
    means = pd.DataFrame(vectors[shown]).groupby(cats[shown]).mean()
    overall = vectors[shown].mean(axis=0)
    centre = means.reindex(cats).to_numpy().copy()
    centre[~np.isfinite(centre).all(axis=1)] = overall
    return vectors - centre


def top_k_distinct(imp_codes, scores, labels, k, seed=42):
    """Per impression: number of distinct `labels` among its top min(k, n) candidates by score (ties in a seeded random order).
    imp_codes: 0..n_imp-1 per candidate row."""
    imp_codes = np.asarray(imp_codes)
    tie = np.random.default_rng(seed).random(len(imp_codes))
    order = np.lexsort((tie, -np.asarray(scores, dtype=np.float64), imp_codes))
    g = imp_codes[order]
    start = np.r_[0, np.cumsum(np.bincount(imp_codes))[:-1]]
    keep = order[(np.arange(len(order)) - start[g]) < k]
    t = pd.DataFrame({"g": imp_codes[keep], "v": np.asarray(labels)[keep]})
    return t.groupby("g")["v"].nunique().reindex(np.arange(imp_codes.max() + 1)).to_numpy()
