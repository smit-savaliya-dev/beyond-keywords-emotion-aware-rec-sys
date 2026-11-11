"""E7 report (plan v5): paired tests of M3b+ and M5b against M2, M3, M4 and M5 on dev (Holm over the 8 pre-declared comparisons), all metrics,
user-level check, shuffled-label control, and the dev subgroups of M3b+ (drift for E3, consistency, clicked category, history length),
all read from the single saved dev prediction file of each model. Plus the E7 charts.

Reads:  saved valid and dev scores and per-impression files of M2 to M5, M3b+ and M5b; behaviours; news; E1, E3 and E7 tables.
Writes: results/tables/behaviour_e7_*.csv; archive/figures/v5_behaviour/e7_*.png.
Not a pipeline step: it reads saved dev result files. It chooses nothing.
Run:    python scripts/06_09_e7_final_report.py
"""
import json

import numpy as np
import pandas as pd
from scipy import stats as st

import _bootstrap  # noqa: F401
from src import behaviour_charts as bc
from src.behaviour import assign_tertile, drift_scores, history_consistency, window_profiles
from src.behaviour_runs import load_context, metrics
from src.config import (BEHAVIOUR_FIGS, BEHAVIOUR_SPEC_VERSION, M2_NAME, M3_NAME, M3BP_NAME, M4_NAME, M5_NAME, M5B_NAME, PROCESSED,
                        RESULTS, SEED, TABLES)
from src.emotion_spec import load_spec
from src.results_tables import compare_pairs
from src.subgroups import group_gains, paired_family, shuffled_labels
from src.where_it_helps import HISTORY_BINS, clicked_category, history_bin, top_categories
from src.run_registry import load_rows

LABELS = {M2_NAME: "M2 Sentence-BERT", M3_NAME: "M3 = M2 + emotion", M3BP_NAME: "M3b+ (emotion, 50 items, no decay)", M4_NAME: "M4 LightGBM",
          M5_NAME: "M5 = M4 + emotion", M5B_NAME: "M5b (LightGBM + emotion, V3)"}
PAIRS = [(a, b) for b in (M3BP_NAME, M5B_NAME) for a in (M2_NAME, M3_NAME, M4_NAME, M5_NAME)]


def per(model, part):
    return pd.read_parquet(RESULTS / "per_impression" / f"{model}__{part}.parquet")


def main():
    spec = load_spec(BEHAVIOUR_SPEC_VERSION)["behaviour"]
    fams, comps, users_lvl = [], [], []
    for part in ("valid", "dev"):
        ctx = load_context(part, with_m2=False)
        users = ctx.users_by_imp
        fams.append(paired_family([(f"{LABELS[b]} - {LABELS[a]}", per(a, part), per(b, part)) for a, b in PAIRS], users).assign(part=part))
        comp, ul = compare_pairs(RESULTS / "per_impression", PAIRS, part)
        comps.append(comp)
        users_lvl.append(ul)
        if part == "dev":
            dev_ctx = ctx
    fam, comp, ul = pd.concat(fams, ignore_index=True), pd.concat(comps, ignore_index=True), pd.concat(users_lvl, ignore_index=True)
    fam.to_csv(TABLES / "behaviour_e7_final_family.csv", index=False)
    comp.to_csv(TABLES / "behaviour_e7_final_all_metrics.csv", index=False)
    ul.to_csv(TABLES / "behaviour_e7_final_user_level.csv", index=False)
    print(fam[fam["part"] == "dev"].round(5).to_string(index=False))

    ctx = dev_ctx
    users = ctx.users_by_imp
    keys = pd.MultiIndex.from_frame(ctx.truth[["imp_key", "news_id"]])
    sc = {m: pd.read_parquet(RESULTS / "scores" / f"{m}__dev.parquet").set_index(["imp_key", "news_id"])["score"].reindex(keys).to_numpy()
          for m in LABELS}
    sh = shuffled_labels(ctx.truth, SEED)["label"].to_numpy()
    q = {m: metrics(ctx, s, sh) for m, s in sc.items()}
    ctrl = paired_family([(f"{LABELS[b]} - {LABELS[a]}, labels shuffled", q[a], q[b]) for a, b in ((M3_NAME, M3BP_NAME), (M4_NAME, M5B_NAME))], users)
    ctrl["auc_shuffled_model_b"] = [float(np.nanmean(q[b]["auc"])) for b in (M3BP_NAME, M5B_NAME)]
    ctrl.to_csv(TABLES / "behaviour_e7_dev_shuffled_control.csv", index=False)

    # dev subgroups of M3b+ from its single prediction file
    p2, p3, pb = per(M2_NAME, "dev"), per(M3_NAME, "dev"), per(M3BP_NAME, "dev")
    rows = ctx.imps["prof_row"].to_numpy()
    e1 = json.loads((TABLES / "behaviour_e1_meta.json").read_text(encoding="utf-8"))
    e3 = json.loads((TABLES / "behaviour_e3_meta.json").read_text(encoding="utf-8"))
    pair, _ = history_consistency(ctx.hist_rows, ctx.vectors)
    d = spec["drift"]
    drift = drift_scores(ctx.hist_rows, ctx.vectors, d["recent"], d["older_max"], d["min_items"])
    _, used = window_profiles(ctx.hist_rows, ctx.vectors, 50, 0.9)
    news = pd.read_parquet(PROCESSED / "news.parquet", columns=["news_id", "category"])
    category_of = news.set_index("news_id")["category"].astype(str)
    fit_clicks = load_rows(PROCESSED, ["fit"], ["news_id", "label"])
    top = top_categories(fit_clicks.loc[fit_clicks["label"] == 1, "news_id"].astype(str).to_numpy(), category_of)
    idx = ctx.imps["imp_key"].to_numpy()
    splits = {"consistency tertile (E1)": pd.Series(assign_tertile(pair[rows], e1["cuts_fit"]), index=idx),
              "drift tertile (E3)": pd.Series(assign_tertile(drift[rows], e3["cuts_fit"], missing="under_10_items"), index=idx),
              "clicked category": clicked_category(ctx.truth, category_of, top),
              "known history items": pd.Series(history_bin(used[rows]), index=idx)}
    sub = []
    for name, groups in splits.items():
        for a, la, b, lb in ((p3, "M3", pb, "M3b+"), (p2, "M2", pb, "M3b+"), (p2, "M2", p3, "M3")):
            sub.append(group_gains(a, b, groups, users, metrics=("auc",), name_a=la, name_b=lb).assign(split=name, part="dev"))
    sub = pd.concat(sub, ignore_index=True)
    sub.to_csv(TABLES / "behaviour_e7_m3bp_dev_subgroups.csv", index=False)
    dt = sub[(sub["split"] == "drift tertile (E3)") & (sub["model_a"] == "M3") & (sub["model_b"] == "M3b+")]
    g = pb.set_index("imp_key")["auc"].reindex(idx).to_numpy() - p3.set_index("imp_key")["auc"].reindex(idx).to_numpy()
    lo_hi = splits["drift tertile (E3)"].to_numpy()
    t = st.ttest_ind(g[(lo_hi == "high") & np.isfinite(g)], g[(lo_hi == "low") & np.isfinite(g)], equal_var=False)
    pd.DataFrame([{"part": "dev", "pair": "M3b+ - M3", "high_minus_low": float(np.nanmean(g[lo_hi == "high"]) - np.nanmean(g[lo_hi == "low"])),
                   "p": float(t.pvalue)}]).to_csv(TABLES / "behaviour_e3_dev_trend.csv", index=False)

    # charts
    v = pd.read_csv(TABLES / "behaviour_e7_m5b_variants.csv")
    refs = {"V1": "ref_M4", "V2": "ref_M4_exposure", "V3": "ref_M4_exposure", "V4": "ref_M4_exposure_regularised", "V5": "ref_M4_exposure_regularised"}
    var = list(refs)
    agg = v.groupby("variant")["auc"].agg(["mean", "min", "max"])
    meta = json.loads((TABLES / "behaviour_e7_m5b_meta.json").read_text(encoding="utf-8"))
    bc.grouped(BEHAVIOUR_FIGS / "e7_m5b_variants_valid.png", "Which LightGBM + emotion variant works best?",
               f"Valid AUC of the five pre-declared M5b variants and their no-emotion references (mean of 3 seeds, bars = min to max). Chosen: {meta['chosen']}. "
               "Valid chose it, so it is optimistic.",
               var, [("reference (no emotion)", [agg.loc[refs[x], "mean"] for x in var], [agg.loc[refs[x], "min"] for x in var], [agg.loc[refs[x], "max"] for x in var], bc.BLUE),
                     ("with emotion", [agg.loc[x, "mean"] for x in var], [agg.loc[x, "min"] for x in var], [agg.loc[x, "max"] for x in var], bc.ORANGE)],
               "AUC (valid)", fmt="{:.4f}", ylim=(float(agg["min"].min()) - 0.004, float(agg["max"].max()) + 0.003), legend_loc="upper left")
    order = [M2_NAME, M3_NAME, M3BP_NAME, M4_NAME, M5_NAME, M5B_NAME]
    vals, lo, hi = [], [], []
    for m in order:
        a = per(m, "dev")["auc"].dropna().to_numpy()
        half = st.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
        vals.append(a.mean())
        lo.append(a.mean() - half)
        hi.append(a.mean() + half)
    f = fam[(fam["part"] == "dev")].set_index("comparison")
    s1 = f.loc[f"{LABELS[M3BP_NAME]} - {LABELS[M3_NAME]}"]
    s2 = f.loc[f"{LABELS[M5B_NAME]} - {LABELS[M4_NAME]}"]
    bc.bars(BEHAVIOUR_FIGS / "e7_final_comparison.png", "Final test: emotion models against content-only ranking",
            f"Final-test AUC with 95% CIs. M3b+ - M3 {s1['mean_diff']:+.4f} (Holm p {s1['p_holm']:.3g}); M5b - M4 {s2['mean_diff']:+.4f} "
            f"(Holm p {s2['p_holm']:.3g}). Orange = with emotion.",
            ["M2\ncontent only", "M3\n+ emotion", "M3b+\n+ emotion, no decay", "M4\nLightGBM", "M5\n+ emotion", "M5b\n+ emotion (V3)"], vals, "AUC (final test)", low=lo, high=hi, fmt="{:.4f}",
            colors=[bc.ORANGE if m in (M3_NAME, M3BP_NAME, M5_NAME, M5B_NAME) else bc.BLUE for m in order], zero=False, base=0.5,
            lim=(0.5, max(hi) + 0.015))
    imp = pd.read_csv(TABLES / "behaviour_e7_m5b_importance.csv").sort_values("contrib_share", ascending=True).tail(16)
    fam_col = {"popularity": bc.PURPLE, "content": bc.BLUE, "context": bc.GREY, "emotion": bc.ORANGE}
    fam_of = {f: ("emotion" if f in ("consistency", "cong_short_personal") else ("context" if f == "exposure_bucket" else None)) for f in imp["feature"]}
    imp["family2"] = [fam_of[f] or fm for f, fm in zip(imp["feature"], imp["family"])]
    em = imp.loc[imp["family2"] == "emotion", "contrib_share"].sum()
    bc.bars(BEHAVIOUR_FIGS / "e7_m5b_feature_importance.png", "What the final M5b relies on",
            f"Share of the mean absolute contribution to the M5b score per feature (valid, seed 42). Orange = emotion features ({100 * em:.1f}% in total), "
            "blue = content, purple = popularity, grey = context.",
            imp["feature"].tolist(), (100 * imp["contrib_share"]).tolist(), "Share of contribution (%)", fmt="{:.1f}%", zero=False,
            colors=[fam_col[f] for f in imp["family2"]], horizontal=True)
    for split, fname, title in (("clicked category", "e7_m3bp_gain_by_category.png", "Where M3b+ helps: by the category of the clicked news"),
                                ("known history items", "e7_m3bp_gain_by_history.png", "Where M3b+ helps: by how much history the user has")):
        x = sub[(sub["split"] == split) & (sub["model_a"] == "M2") & (sub["model_b"] == "M3b+")]
        if split == "known history items":
            x = x.set_index("group").reindex([b[2] for b in HISTORY_BINS]).dropna(subset=["gain"]).reset_index()
        else:
            x = x.sort_values("gain")
        bc.bars(BEHAVIOUR_FIGS / fname, title,
                "Descriptive: final-test AUC gain of M3b+ over M2 (content only) per group, from the single dev prediction files; 95% user-bootstrap CIs. "
                "Groups were fixed before the dev run.",
                [str(gr) for gr in x["group"]], x["gain"].tolist(), "AUC gain over M2 (final test)", low=x["ci_low"].tolist(), high=x["ci_high"].tolist(),
                colors=[bc.ORANGE if gv > 0 else bc.GREY for gv in x["gain"]], horizontal=(split == "clicked category"),
                notes=None if split == "clicked category" else [f"n = {int(n):,}" for n in x["impressions"]])


if __name__ == "__main__":
    main()
