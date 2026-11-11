"""E1 (plan v5): do emotionally consistent users gain more from emotion? Consistency tertiles fixed on fit, gains of M3 over M2 on valid
and dev (from the saved prediction files), trend tests, and M3b: alpha chosen per tertile on valid (valid only).

Reads:  behaviours and impressions of fit, valid and dev; saved M2 and M3 scores and per-impression files (valid, dev); spec v5.
Writes: results/tables/behaviour_e1_*.csv and behaviour_e1_meta.json; archive/figures/v5_behaviour/e1_*.png.
Not a pipeline step: it reads saved dev result files (like 04_08). Selects only on valid; dev is reported, never used to choose.
Run:    python scripts/06_01_e1_consistency.py
"""
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import behaviour_charts as bc
from src.behaviour import assign_tertile, history_consistency, tertile_cuts
from src.behaviour_runs import emotion_score, fused_scores, load_context, metrics
from src.config import BEHAVIOUR_FIGS, BEHAVIOUR_SPEC_VERSION, M2_NAME, M3_NAME, RESULTS, SEED, TABLES
from src.emotion_spec import load_spec
from src.provenance import run_info
from src.stats import holm
from src.subgroups import group_gains, group_means, paired_family, shuffled_labels, trend_tests

ORDER = ["low", "mid", "high", "no_history"]


def consistency(ctx):
    """(primary, secondary) consistency per impression, from the history column only."""
    pair, cent = history_consistency(ctx.hist_rows, ctx.vectors)
    rows = ctx.imps["prof_row"].to_numpy()
    return pair[rows], cent[rows]


def saved_per(model, part):
    return pd.read_parquet(RESULTS / "per_impression" / f"{model}__{part}.parquet")


def main():
    spec = load_spec(BEHAVIOUR_SPEC_VERSION)["behaviour"]
    fit = load_context("fit", with_m2=False)
    c_fit, _ = consistency(fit)
    cuts = tertile_cuts(c_fit)
    print(f"fit cut-offs {cuts[0]:.4f} / {cuts[1]:.4f}; {np.isfinite(c_fit).mean():.1%} of fit impressions have 2+ items")
    counts, edges = np.histogram(c_fit[np.isfinite(c_fit)], bins=50)
    pd.DataFrame({"bin_low": edges[:-1], "bin_high": edges[1:], "impressions": counts}).to_csv(TABLES / "behaviour_e1_consistency_fit.csv", index=False)

    gains, means, trends, controls = [], [], [], []
    ctxs = {}
    for part in ("valid", "dev"):
        ctx = load_context(part, with_m2=(part == "valid"))
        ctxs[part] = ctx
        c, c2 = consistency(ctx)
        groups = pd.Series(assign_tertile(c, cuts), index=ctx.imps["imp_key"].to_numpy())
        users = ctx.users_by_imp
        p2, p3 = saved_per(M2_NAME, part), saved_per(M3_NAME, part)
        gains.append(group_gains(p2, p3, groups, users, name_a="M2", name_b="M3").assign(part=part))
        means += [group_means(p2, groups, users, name="M2").assign(part=part), group_means(p3, groups, users, name="M3").assign(part=part)]
        g = p3.set_index("imp_key")["auc"].reindex(ctx.imps["imp_key"]).to_numpy() - p2.set_index("imp_key")["auc"].reindex(ctx.imps["imp_key"]).to_numpy()
        prim = trend_tests(c, g, groups.to_numpy())
        sec = trend_tests(c2, g, assign_tertile(c2, tertile_cuts(consistency(fit)[1])))
        prim["p_holm_rho"], prim["p_holm_hl"] = holm([prim["rho_p"], prim["hl_p"]])
        trends += [{"part": part, "score": "pairwise (primary)", **prim}, {"part": part, "score": "centroid (secondary)", **sec}]
        # controls: consistency permuted across impressions; labels permuted within impressions
        perm = np.random.default_rng(SEED).permutation(len(c))
        controls.append({"part": part, "control": "consistency permuted", **trend_tests(c[perm], g, groups.to_numpy()[perm])})
        sh = shuffled_labels(ctx.truth, SEED)["label"].to_numpy()
        s2 = pd.read_parquet(RESULTS / "scores" / f"{M2_NAME}__{part}.parquet").set_index(["imp_key", "news_id"])["score"]
        s3 = pd.read_parquet(RESULTS / "scores" / f"{M3_NAME}__{part}.parquet").set_index(["imp_key", "news_id"])["score"]
        keys = pd.MultiIndex.from_frame(ctx.truth[["imp_key", "news_id"]])
        q2, q3 = metrics(ctx, s2.reindex(keys).to_numpy(), sh), metrics(ctx, s3.reindex(keys).to_numpy(), sh)
        gsh = q3["auc"].to_numpy() - q2["auc"].to_numpy()
        controls.append({"part": part, "control": "labels shuffled", "mean_auc_m3": float(np.nanmean(q3["auc"])),
                         **trend_tests(c, gsh, groups.to_numpy())})
    gains, means = pd.concat(gains, ignore_index=True), pd.concat(means, ignore_index=True)
    trends, controls = pd.DataFrame(trends), pd.DataFrame(controls)
    gains.to_csv(TABLES / "behaviour_e1_gain_by_tertile.csv", index=False)
    means.to_csv(TABLES / "behaviour_e1_auc_by_tertile.csv", index=False)
    trends.to_csv(TABLES / "behaviour_e1_trend.csv", index=False)
    controls.to_csv(TABLES / "behaviour_e1_controls.csv", index=False)
    print(gains[gains["metric"] == "auc"][["part", "group", "impressions", "gain", "ci_low", "ci_high", "too_small"]].round(5).to_string(index=False))
    print(trends.round(5).to_string(index=False))

    # M3b: alpha per tertile, chosen on valid only
    ctx = ctxs["valid"]
    emo = emotion_score(ctx)
    c, _ = consistency(ctx)
    tert = assign_tertile(c, cuts)
    users = ctx.users_by_imp
    m3_fast = metrics(ctx, fused_scores(ctx, emo, 0.8))
    m2_fast = metrics(ctx, ctx.m2)
    repro = float(np.nanmax(np.abs(m3_fast["auc"].to_numpy() - saved_per(M3_NAME, "valid").set_index("imp_key")["auc"].reindex(m3_fast["imp_key"]).to_numpy())))
    grid = []
    per_alpha = {}
    for a in spec["adaptive_alpha_grid"]:
        per_alpha[a] = metrics(ctx, fused_scores(ctx, emo, a))
        for t in ORDER:
            grid.append({"alpha": a, "group": t, "valid_auc": float(np.nanmean(per_alpha[a]["auc"].to_numpy()[tert == t]))})
    grid = pd.DataFrame(grid)
    chosen = {}
    for t in ORDER[:3]:
        g = grid[grid["group"] == t]
        best = g["valid_auc"].max()
        chosen[t] = float(g.loc[g["valid_auc"] == best, "alpha"].iloc[np.argmin(np.abs(g.loc[g["valid_auc"] == best, "alpha"] - 0.8))])
    chosen["no_history"] = 0.8
    alpha_imp = np.array([chosen[t] for t in tert])
    m3b = metrics(ctx, fused_scores(ctx, emo, None, alpha_imp))
    fam = paired_family([("M3b - M3", m3_fast, m3b), ("M3b - M2", m2_fast, m3b)], users)
    sh = shuffled_labels(ctx.truth, SEED)["label"].to_numpy()
    ctrl = paired_family([("M3b - M2, labels shuffled", metrics(ctx, ctx.m2, sh), metrics(ctx, fused_scores(ctx, emo, None, alpha_imp), sh))], users)
    fam = pd.concat([fam, ctrl.assign(family="control")], ignore_index=True)
    grid.to_csv(TABLES / "behaviour_e1_m3b_alpha_grid.csv", index=False)
    fam.to_csv(TABLES / "behaviour_e1_m3b_compare.csv", index=False)
    print(fam.round(5).to_string(index=False))
    meta = {"cuts_fit": cuts, "chosen_alpha_valid": chosen, "m3_reproduction_max_abs_auc_diff": repro, "run_info": run_info()}
    (TABLES / "behaviour_e1_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # charts
    a = gains[gains["metric"] == "auc"].set_index(["part", "group"])
    lab = [f"{t.replace('_', ' ')}" for t in ORDER]
    notes = [f"n = {int(a.loc[('dev', t), 'impressions']):,}" for t in ORDER]
    s = [(p, [a.loc[(p, t), "gain"] for t in ORDER], [a.loc[(p, t), "ci_low"] for t in ORDER], [a.loc[(p, t), "ci_high"] for t in ORDER], col)
         for p, col in (("valid", bc.BLUE), ("dev", bc.ORANGE))]
    tr = trends.set_index(["part", "score"]).loc[("dev", "pairwise (primary)")]
    bc.grouped(BEHAVIOUR_FIGS / "e1_gain_by_tertile.png", "Does emotion help consistent users more?",
               f"Gain in AUC from adding emotion (M3 minus M2) per consistency tertile, 95% user-bootstrap CIs. Dev: high minus low "
               f"{tr['high_minus_low']:+.4f} (Holm p {tr['p_holm_hl']:.3g}); Spearman rho {tr['rho']:+.3f}.",
               lab, s, "AUC gain (M3 minus M2)", fmt="{:+.4f}", zero=True, notes=notes)
    m = means[(means["metric"] == "auc") & (means["part"] == "dev")].set_index(["model", "group"])
    s = [(name, [m.loc[(mod, t), "mean"] for t in ORDER], [m.loc[(mod, t), "ci_low"] for t in ORDER], [m.loc[(mod, t), "ci_high"] for t in ORDER], col)
         for mod, name, col in (("M2", "M2 (content only)", bc.BLUE), ("M3", "M3 (content + emotion)", bc.ORANGE))]
    lo_all = min(min(x[2]) for x in s)
    bc.grouped(BEHAVIOUR_FIGS / "e1_auc_by_tertile.png", "Ranking quality by how consistent a user's emotions are",
               "Final-test AUC of M2 and M3 per consistency tertile (from the saved prediction files); bars show 95% user-bootstrap CIs.",
               lab, s, "AUC (final test)", fmt="{:.3f}", ylim=(lo_all - 0.02, None), notes=notes, legend_loc="upper left")
    hist = pd.read_csv(TABLES / "behaviour_e1_consistency_fit.csv")
    vals = np.repeat((hist["bin_low"] + hist["bin_high"]).to_numpy() / 2, hist["impressions"].to_numpy())
    bc.histogram(BEHAVIOUR_FIGS / "e1_consistency_hist.png", "How consistent are users' emotions?",
                 f"Mean pairwise similarity of the emotion vectors of a user's last 50 history items (fit impressions). Dashed lines: the tertile cut-offs "
                 f"{cuts[0]:.3f} and {cuts[1]:.3f}, fixed on fit.", vals, "Emotional consistency (mean pairwise cosine)", cuts=list(cuts),
                 cut_labels=["low | mid", "mid | high"], bins=50)


if __name__ == "__main__":
    main()
