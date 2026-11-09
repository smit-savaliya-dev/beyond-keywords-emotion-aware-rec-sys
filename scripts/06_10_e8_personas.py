"""E8 (plan v5): emotional personas. k-means on users' long-term emotion profiles (fit users, k by silhouette), named by their top emotions,
stability checked on valid users, and the gains of M3 over M2 and M3b+ over M3 per persona on valid and dev (from the saved files).

Reads:  fit, valid and dev behaviours; saved per-impression files of M2, M3 and M3b+; spec v5.
Writes: results/tables/behaviour_e8_*.csv, behaviour_e8_meta.json; results/figures/v5_behaviour/e8_*.png.
Not a pipeline step: it reads saved dev result files. Descriptive, nothing is chosen from the gains.
Run:    python scripts/06_10_e8_personas.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import _bootstrap  # noqa: F401
from src import behaviour_charts as bc
from src.behaviour import fit_personas, match_clusters, persona_names, window_profiles
from src.behaviour_runs import load_context
from src.config import BEHAVIOUR_FIGS, BEHAVIOUR_SPEC_VERSION, M2_NAME, M3_NAME, M3BP_NAME, RESULTS, SEED, TABLES
from src.emotion_features import GO_EMOTIONS
from src.emotion_spec import load_spec
from src.subgroups import group_gains


def user_vectors(ctx, min_items):
    """Long-term profile per profile row (last 50, decay 0.9) and the rows with enough known history."""
    prof, used = window_profiles(ctx.hist_rows, ctx.vectors, 50, 0.9)
    return prof, used >= min_items


def main():
    spec = load_spec(BEHAVIOUR_SPEC_VERSION)["behaviour"]["personas"]
    fit = load_context("fit", with_m2=False)
    prof, ok = user_vectors(fit, spec["min_items"])
    first = pd.Series(np.arange(len(fit.prof_users))).groupby(fit.prof_users).first().to_numpy()   # one profile per user
    rows = first[ok[first]]
    x = prof[rows]
    mu, sd = x.mean(axis=0), x.std(axis=0)
    z = (x - mu) / sd
    k, sil, models = fit_personas(z, spec["k"], SEED, spec["n_init"], spec["silhouette_sample"])
    km = models[k]
    names = persona_names(km.cluster_centers_, GO_EMOTIONS)
    names = [f"P{i + 1}: {n}" for i, n in enumerate(names)]
    print(f"k = {k}, silhouettes {sil}")
    del fit

    gains, sizes = [], []
    stability = {}
    for part in ("valid", "dev"):
        ctx = load_context(part, with_m2=False)
        p, okp = user_vectors(ctx, spec["min_items"])
        zp = (p - mu) / sd
        lab = np.where(okp, km.predict(zp), -1)
        persona = np.array([names[i] if i >= 0 else "too little history" for i in lab], dtype=object)
        if part == "valid":
            f = pd.Series(np.arange(len(ctx.prof_users))).groupby(ctx.prof_users).first().to_numpy()
            f = f[okp[f]]
            refit = KMeans(n_clusters=k, n_init=spec["n_init"], random_state=SEED).fit(zp[f])
            order = match_clusters(km.cluster_centers_, refit.cluster_centers_)
            stability = {"ari_fit_centroids_vs_valid_refit": float(adjusted_rand_score(km.predict(zp[f]), refit.labels_)),
                         "centroid_distance_after_matching": [float(np.linalg.norm(km.cluster_centers_[i] - refit.cluster_centers_[order[i]])) for i in range(k)]}
        imp_persona = pd.Series(persona[ctx.imps["prof_row"].to_numpy()], index=ctx.imps["imp_key"].to_numpy())
        sizes.append(imp_persona.value_counts().rename("impressions").to_frame().assign(part=part).reset_index(names="persona"))
        users = ctx.users_by_imp
        per = {m: pd.read_parquet(RESULTS / "per_impression" / f"{m}__{part}.parquet") for m in (M2_NAME, M3_NAME, M3BP_NAME)}
        gains.append(group_gains(per[M2_NAME], per[M3_NAME], imp_persona, users, metrics=("auc",), name_a="M2", name_b="M3").assign(part=part))
        gains.append(group_gains(per[M3_NAME], per[M3BP_NAME], imp_persona, users, metrics=("auc",), name_a="M3", name_b="M3b+").assign(part=part))
        if part == "valid":
            map_z, map_lab = zp[f], km.predict(zp[f])
    gains, sizes = pd.concat(gains, ignore_index=True), pd.concat(sizes, ignore_index=True)
    centroids = pd.DataFrame(km.cluster_centers_, columns=GO_EMOTIONS, index=names)
    gains.to_csv(TABLES / "behaviour_e8_gain_by_persona.csv", index=False)
    sizes.to_csv(TABLES / "behaviour_e8_persona_sizes.csv", index=False)
    centroids.to_csv(TABLES / "behaviour_e8_persona_profiles.csv")
    meta = {"k": k, "silhouette": sil, "names": names, "fit_users": int(len(rows)), "stability": stability, "standardisation": "fit mean and SD"}
    (TABLES / "behaviour_e8_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(gains.round(5).to_string(index=False), json.dumps(stability, indent=1), sep="\n")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=SEED).fit(z)
    xy = pca.transform(map_z)
    bc.scatter_map(BEHAVIOUR_FIGS / "e8_persona_map.png", "Emotional personas of news readers",
                   f"Valid users placed by the two main directions of their emotion profiles (PCA fitted on fit users), coloured by persona (k = {k}, "
                   f"silhouette {sil[k]:.2f}; stability ARI {stability['ari_fit_centroids_vs_valid_refit']:.2f}). X = persona centre.",
                   xy[:, 0], xy[:, 1], map_lab, names, f"Direction 1 ({100 * pca.explained_variance_ratio_[0]:.0f}% of variation)",
                   f"Direction 2 ({100 * pca.explained_variance_ratio_[1]:.0f}%)")
    s = sizes[sizes["part"] == "dev"].set_index("persona").reindex([*names, "too little history"])
    bc.bars(BEHAVIOUR_FIGS / "e8_persona_sizes.png", "How common is each persona?",
            "Share of final-test impressions per persona (users with fewer than 5 known history items are 'too little history').",
            [n.replace(": ", ":\n") for n in s.index], (100 * s["impressions"] / s["impressions"].sum()).tolist(), "Impressions (%)", fmt="{:.1f}%",
            colors=[*bc.PALETTE[:k], bc.GREY], zero=False)
    bc.heatmap(BEHAVIOUR_FIGS / "e8_persona_profiles.png", "What each persona reads",
               "Persona centres: how far each emotion lies above (orange) or below (blue) the average user, in standard deviations (fit users).",
               centroids.to_numpy(), names, GO_EMOTIONS, "Emotion", "Persona", fmt="{:+.2f}", center=0.0)
    g = gains[gains["part"] == "dev"].set_index(["model_a", "model_b", "group"])
    groups = [*names, "too little history"]
    bc.grouped(BEHAVIOUR_FIGS / "e8_gain_by_persona.png", "Does emotion help some personas more?",
               "Final-test AUC gain per persona: M3 over M2 (emotion added to content) and M3b+ over M3 (refined profile), 95% user-bootstrap CIs.",
               [n.split(": ")[0] for n in groups],
               [(lab, [g.loc[(a, b, n), "gain"] for n in groups], [g.loc[(a, b, n), "ci_low"] for n in groups], [g.loc[(a, b, n), "ci_high"] for n in groups], col)
                for a, b, lab, col in (("M2", "M3", "M3 - M2", bc.BLUE), ("M3", "M3b+", "M3b+ - M3", bc.ORANGE))],
               "AUC gain (final test)", fmt="{:+.4f}", zero=True, notes=[n.split(": ")[1] if ": " in n else "" for n in groups])


if __name__ == "__main__":
    main()
