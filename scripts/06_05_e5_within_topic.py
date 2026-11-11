"""E5 (plan v5): is emotion just topic in disguise? (a) AUC of the personal emotion match among candidates of the same category (fit, valid),
and M3 - M2 within category (valid, dev, from the saved files). (b) Residual emotion (item minus its category mean, fit items): does it add to
M2 fused with the user's topic share? Weights chosen on valid.

Reads:  fit, valid and dev behaviours and impressions; news categories; saved M2 and M3 scores; the valid M4 feature table (user_cat_affinity).
Writes: results/tables/behaviour_e5_*.csv; archive/figures/v5_behaviour/e5_*.png.
Not a pipeline step: it reads saved dev result files. Only valid chooses.
Run:    python scripts/06_05_e5_within_topic.py
"""
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import behaviour_charts as bc
from src.behaviour import residual_vectors
from src.behaviour_runs import emotion_score, fused_scores, load_context, metrics
from src.config import BEHAVIOUR_FIGS, BEHAVIOUR_SPEC_VERSION, LTR_FEATURE_FILE, M2_NAME, M3_NAME, PROCESSED, RESULTS, SEED, TABLES
from src.congruence import cosine_rows, user_profiles
from src.emotion_spec import load_spec
from src.fusion import fuse, minmax_within
from src.stats import holm, paired_test
from src.subgroups import paired_family, shuffled_labels, user_bootstrap_ci, within_group_metrics

TOP = 10


def within_category(ctx, scores, cats, labels=None):
    """Per (impression, category) AUC of `scores` among candidates of that category. -> table imp_key, category, auc."""
    y = ctx.truth["label"].to_numpy() if labels is None else labels
    t = within_group_metrics(ctx.truth["imp_key"].to_numpy(), cats[ctx.item_idx], y, scores)
    return t.rename(columns={"group": "category"})[["imp_key", "category", "auc"]]


def summarise(t, users, top, value="auc", ref=0.5):
    """Mean per category (top categories, rest 'other') and overall, with user-bootstrap CIs and a t-test against `ref`."""
    t = t.dropna(subset=[value]).copy()
    t["group"] = np.where(t["category"].isin(top), t["category"], "other")
    t["user"] = t["imp_key"].map(users)
    rows = []
    for g, d in [*t.groupby("group"), ("all", t)]:
        lo, hi = user_bootstrap_ci(d[value].to_numpy(), d["user"].to_numpy())
        tt = paired_test(np.full(len(d), ref), d[value].to_numpy())
        rows.append({"group": g, "n": len(d), "mean": float(d[value].mean()), "ci_low": lo, "ci_high": hi, "p_vs_ref": tt["p"], "too_small": len(d) < 500})
    return pd.DataFrame(rows)


def main():
    grid = load_spec(BEHAVIOUR_SPEC_VERSION)["behaviour"]["residual"]["fusion_grid"]
    news = pd.read_parquet(PROCESSED / "news.parquet", columns=["news_id", "category"])
    fit = load_context("fit", with_m2=False)
    cats = news.set_index("news_id")["category"].astype(str).reindex(fit.news_ids).to_numpy()
    clicked = fit.item_idx[fit.truth["label"].to_numpy() == 1]
    top = pd.Series(cats[clicked]).value_counts().head(TOP).index.tolist()
    resid = residual_vectors(fit.vectors, cats, fit.item_idx)

    rows_a, gains_cat = [], []
    t_fit = within_category(fit, emotion_score(fit), cats)
    rows_a.append(summarise(t_fit, fit.users_by_imp, top).assign(part="fit", score="personal emotion match alone"))
    del fit
    for part in ("valid", "dev"):
        ctx = load_context(part, with_m2=(part == "valid"))
        users = ctx.users_by_imp
        keys = pd.MultiIndex.from_frame(ctx.truth[["imp_key", "news_id"]])
        s = {m: pd.read_parquet(RESULTS / "scores" / f"{m}__{part}.parquet").set_index(["imp_key", "news_id"])["score"].reindex(keys).to_numpy()
             for m in (M2_NAME, M3_NAME)}
        w2, w3 = within_category(ctx, s[M2_NAME], cats), within_category(ctx, s[M3_NAME], cats)
        g = w3.merge(w2, on=["imp_key", "category"], suffixes=("", "_m2"))
        g["gain"] = g["auc"] - g["auc_m2"]
        gains_cat.append(summarise(g.drop(columns="auc").rename(columns={"gain": "auc"}), users, top, ref=0.0).assign(part=part))
        if part == "valid":
            emo = emotion_score(ctx)
            rows_a.append(summarise(within_category(ctx, emo, cats), users, top).assign(part=part, score="personal emotion match alone"))
            sh = shuffled_labels(ctx.truth, SEED)["label"].to_numpy()
            rows_a.append(summarise(within_category(ctx, emo, cats, sh), users, top).assign(part=part, score="labels shuffled (control)"))
            # (b) residual emotion on top of M2 + topic
            aff = pd.read_parquet(PROCESSED / LTR_FEATURE_FILE.format(part="valid"), columns=["imp_key", "user_cat_affinity"])
            if aff["imp_key"].tolist() != ctx.truth["imp_key"].tolist():
                raise ValueError("feature rows are not in harness order")
            imps = ctx.imp_codes
            base, topic = minmax_within(imps, ctx.m2), minmax_within(imps, aff["user_cat_affinity"].to_numpy(np.float64))
            topic_grid = [(a, metrics(ctx, fuse(base, topic, a))) for a in grid]
            a_best, per_topic = max(topic_grid, key=lambda r: (np.nanmean(r[1]["auc"]), r[0]))
            topic_scores = fuse(base, topic, a_best)
            prof, _ = user_profiles(ctx.hist_rows, resid, np.zeros(resid.shape[1]), decay=0.9, max_items=50)
            rmatch = cosine_rows(resid[ctx.item_idx], prof[ctx.groups])
            rnorm, tnorm = minmax_within(imps, rmatch), minmax_within(imps, topic_scores)
            resid_grid = [(b, metrics(ctx, fuse(tnorm, rnorm, b))) for b in grid]
            b_best, per_resid = max(resid_grid, key=lambda r: (np.nanmean(r[1]["auc"]), r[0]))
            per_m2, per_m3 = metrics(ctx, ctx.m2), metrics(ctx, fused_scores(ctx, emo, 0.8))
            fam = paired_family([("topic + residual emotion - topic only", per_topic, per_resid)], users)
            ctrl = paired_family([("topic + residual - topic only, labels shuffled", metrics(ctx, topic_scores, sh), metrics(ctx, fuse(tnorm, rnorm, b_best), sh))], users)
            within_all = rows_a[-2].set_index("group").loc["all"]
            fam = pd.concat([fam, pd.DataFrame([{"comparison": "within-category AUC of emotion alone vs 0.5", "metric": "auc", "n": within_all["n"],
                                                 "mean_diff": within_all["mean"] - 0.5, "p": within_all["p_vs_ref"]}])], ignore_index=True)
            fam["p_holm"] = holm(fam["p"].to_numpy())
            fam["significant_holm"] = fam["p_holm"] < 0.05
            fam = pd.concat([fam, ctrl.assign(family="control")], ignore_index=True)
            ladder = pd.DataFrame([{"model": n, "valid_auc": float(np.nanmean(p["auc"]))} for n, p in
                                   (("M2 content only", per_m2), (f"M2 + topic (alpha {a_best})", per_topic),
                                    (f"M2 + topic + residual emotion (beta {b_best})", per_resid), ("M3 = M2 + emotion", per_m3))])
            ladder.to_csv(TABLES / "behaviour_e5_topic_vs_residual.csv", index=False)
            fam.to_csv(TABLES / "behaviour_e5_family.csv", index=False)
            print(ladder.round(5).to_string(index=False), fam.round(5).to_string(index=False), sep="\n")
        del ctx
    a = pd.concat(rows_a, ignore_index=True)
    gcat = pd.concat(gains_cat, ignore_index=True)
    a.to_csv(TABLES / "behaviour_e5_within_category_auc.csv", index=False)
    gcat.to_csv(TABLES / "behaviour_e5_m3_gain_within_category.csv", index=False)
    print(a.round(4).to_string(index=False), gcat.round(5).to_string(index=False), sep="\n")

    v = a[(a["part"] == "valid") & (a["score"] == "personal emotion match alone")].sort_values("mean")
    allv = v.set_index("group").loc["all"]
    v = v[v["group"] != "all"]
    bc.bars(BEHAVIOUR_FIGS / "e5_within_category_auc.png", "Does emotion still rank news within the same topic?",
            f"Valid AUC of the personal emotion match alone among candidates of the same category in an impression (0.5 = random). "
            f"All categories {allv['mean']:.3f} (95% CI {allv['ci_low']:.3f}-{allv['ci_high']:.3f}).",
            v["group"].tolist(), v["mean"].tolist(), "AUC within the category (valid)", low=v["ci_low"].tolist(), high=v["ci_high"].tolist(),
            colors=[bc.ORANGE if x > 0.5 else bc.GREY for x in v["mean"]], fmt="{:.3f}", zero=False, hline=0.5, hline_label="random",
            horizontal=True, base=0.5, lim=(0.44, max(0.62, float(v["ci_high"].max()) + 0.02)))
    lad = pd.read_csv(TABLES / "behaviour_e5_topic_vs_residual.csv")
    bc.bars(BEHAVIOUR_FIGS / "e5_topic_vs_residual.png", "Is emotion just topic in disguise?",
            "Valid AUC: M2, M2 fused with the user's topic share, that plus residual emotion (item emotion minus its category mean), and M3. "
            "Weights were chosen on valid, so all are optimistic.",
            [m.replace(" (", "\n(") for m in lad["model"]], lad["valid_auc"].tolist(), "AUC (valid)", fmt="{:.4f}", zero=False,
            colors=[bc.BLUE, bc.SKY, bc.ORANGE, bc.ORANGE], base=0.5, lim=(0.5, float(lad["valid_auc"].max()) + 0.02))


if __name__ == "__main__":
    main()
