"""Stage 7 (plan v6): the pre-declared families (dev, Holm within each), the fold evidence, and the v6 charts.
Dev numbers are a SECOND LOOK (D79); the primary evidence is the rolling folds (trees) and valid seeds (NRMS).

Reads:  saved valid/dev per-impression files, fold tables (v6_trees_*.csv, ltr_v6_eval_*), NRMS seed tables and curves, selection files.
Writes: results/tables/v6_final_*.csv; archive/figures/v6_improve/*.png and overview_v6.png; archive/figures/v5_behaviour/e7_final_comparison.png.
Not a pipeline step: it reads saved dev result files. Run: python scripts/07_06_final_report.py
"""
import json
import shutil

import numpy as np
import pandas as pd
from scipy import stats as st

import _bootstrap  # noqa: F401
from src import behaviour_charts as bc
from src.behaviour_runs import load_context
from src.config import ARCHIVE_FIGS, BEHAVIOUR_FIGS, FIGURES, FROZEN_DIR, PROCESSED, RESULTS, TABLES
from src.nrms_runs import RUN_DIR
from src.stats import paired_test
from src.subgroups import paired_family, shuffled_labels, fast_metrics

OUT = ARCHIVE_FIGS / "v6_improve"
NRMS = {"M7a": "M7a_nrms_emotion_item", "M7b": "M7b_nrms_emotion_gated", "M7c": "M7c_nrms_emotion_both"}


def per(m, part):
    return pd.read_parquet(RESULTS / "per_impression" / f"{m}__{part}.parquet")


def main():
    sel_t = json.loads((TABLES / "v6_trees_selection.json").read_text(encoding="utf-8"))
    sel_n = json.loads((TABLES / "v6_nrms_selection.json").read_text(encoding="utf-8"))
    m5src = json.loads((FROZEN_DIR / "M5v6.json").read_text(encoding="utf-8"))["config"]["source"]
    m7 = NRMS[sel_n["selected"]]
    ctx = load_context("dev", with_m2=False)
    users = ctx.users_by_imp
    primary = [("M5v6 - M4v6 (trees)", "M4v6", "M5v6"), ("M7 - M6 (NRMS)", "M6_nrms", m7), ("M7 - M7ctrl (emotion information)", "M7ctrl_nrms_permuted", m7),
               ("B1 - B0 (blends)", "B0", "B1")]
    dev_models = {m for _, a, b in primary for m in (a, b)}
    ladder = {m: float(per(m, "dev")["auc"].mean()) for m in ["M4_lightgbm", "M5_lightgbm_emotion", "M3b_plus", *dev_models]}
    best = max(ladder, key=ladder.get)
    secondary = [("M4v6 - M4", "M4_lightgbm", "M4v6"), ("B0 - M4v6", "M4v6", "B0"), ("B1 - M5", "M5_lightgbm_emotion", "B1"), (f"best ({best}) - M4", "M4_lightgbm", best)]
    fam_p = paired_family([(n, per(a, "dev"), per(b, "dev")) for n, a, b in primary], users).assign(family="primary (emotion vs twin)")
    fam_s = paired_family([(n, per(a, "dev"), per(b, "dev")) for n, a, b in secondary], users).assign(family="secondary (best overall)")
    sh = shuffled_labels(ctx.truth, 42)["label"].to_numpy()
    keys = pd.MultiIndex.from_frame(ctx.truth[["imp_key", "news_id"]])
    ctrl = []
    for _, a, b in primary:
        sa, sb = (pd.read_parquet(RESULTS / "scores" / f"{m}__dev.parquet").set_index(["imp_key", "news_id"])["score"].reindex(keys).to_numpy() for m in (a, b))
        qa, qb = (fast_metrics(ctx.truth["imp_key"].to_numpy(), sh, x) for x in (sa, sb))
        t = paired_test(qa["auc"], qb["auc"])
        ctrl.append({"comparison": f"{b} - {a}, labels shuffled", "auc_b": float(np.nanmean(qb["auc"])), "mean_diff": t["mean_diff"], "p": t["p"]})
    fam = pd.concat([fam_p, fam_s], ignore_index=True)
    fam.to_csv(TABLES / "v6_final_dev_families.csv", index=False)
    pd.DataFrame(ctrl).to_csv(TABLES / "v6_final_dev_shuffled_controls.csv", index=False)

    # fold evidence: trees per fold (seed-mean per-impression tables), NRMS valid seeds
    folds = []
    for f in ("F1", "F2", "F3", "F4"):
        a, b = (pd.read_parquet(PROCESSED / f"ltr_v6_eval_{m}_{f}.parquet") for m in ("M4v6", m5src))
        t = paired_test(a["auc"], b["auc"])
        folds.append({"comparison": "M5v6 - M4v6", "fold": f, "gain": t["mean_diff"], "ci_low": t["ci_low"], "ci_high": t["ci_high"], "dz": t["dz"], "p": t["p"]})
    pa, pb = per("M6_nrms", "valid"), per(m7, "valid")
    t = paired_test(pa.set_index("imp_key")["auc"], pb.set_index("imp_key").reindex(pa["imp_key"])["auc"].to_numpy())
    folds.append({"comparison": "M7 - M6", "fold": "F4 (valid)", "gain": t["mean_diff"], "ci_low": t["ci_low"], "ci_high": t["ci_high"], "dz": t["dz"], "p": t["p"]})
    pc = per("M7ctrl_nrms_permuted", "valid")
    t = paired_test(pc.set_index("imp_key")["auc"], pb.set_index("imp_key").reindex(pc["imp_key"])["auc"].to_numpy())
    folds.append({"comparison": "M7 - M7ctrl", "fold": "F4 (valid)", "gain": t["mean_diff"], "ci_low": t["ci_low"], "ci_high": t["ci_high"], "dz": t["dz"], "p": t["p"]})
    ba, bb = per("B0", "valid"), per("B1", "valid")
    t = paired_test(ba["auc"], bb.set_index("imp_key").reindex(ba["imp_key"])["auc"].to_numpy())
    folds.append({"comparison": "B1 - B0", "fold": "F4 (valid)", "gain": t["mean_diff"], "ci_low": t["ci_low"], "ci_high": t["ci_high"], "dz": t["dz"], "p": t["p"]})
    folds = pd.DataFrame(folds)
    folds.to_csv(TABLES / "v6_final_fold_evidence.csv", index=False)
    print(fam.round(5).to_string(index=False), folds.round(5).to_string(index=False), pd.DataFrame(ctrl).round(4).to_string(index=False), sep="\n")

    # charts
    OUT.mkdir(parents=True, exist_ok=True)
    fp = fam_p.set_index("comparison")
    fold_mean = folds.groupby("comparison")["gain"].mean()
    labs = ["Trees\nM5v6 - M4v6", "NRMS\nM7 - M6", "Emotion information\nM7 - M7ctrl", "Blends\nB1 - B0"]
    keymap = {"M5v6 - M4v6 (trees)": "M5v6 - M4v6", "M7 - M6 (NRMS)": "M7 - M6", "M7 - M7ctrl (emotion information)": "M7 - M7ctrl", "B1 - B0 (blends)": "B1 - B0"}
    fl = folds.groupby("comparison")[["ci_low", "ci_high"]].mean()
    bc.grouped(OUT / "fair_twin.png", "Does emotion beat the same model without it?",
               "AUC gain of each emotion model over its no-emotion twin: rolling folds / valid (primary evidence, mean over folds) and the final test "
               "(a second look at dev), 95% CIs, Holm over the four. The blend bar mixes emotion with a different blend weight: at the same weight "
               "B1 - B0 is only about +0.0008 on the final test (post-hoc, v6_posthoc_same_weight_blends.csv).",
               labs, [("folds / valid (primary)", [fold_mean[keymap[k]] for k in fp.index], [fl.loc[keymap[k], "ci_low"] for k in fp.index],
                       [fl.loc[keymap[k], "ci_high"] for k in fp.index], bc.BLUE),
                      ("final test (second look)", fp["mean_diff"].tolist(), fp["ci_low"].tolist(), fp["ci_high"].tolist(), bc.ORANGE)],
               "AUC gain over the twin", fmt="{:+.4f}", zero=True, notes=[f"Holm p {p:.2g}" for p in fp["p_holm"]])
    tf = pd.read_csv(TABLES / "v6_trees_folds.csv")
    m = tf.groupby(["model", "fold"])["auc"].mean().unstack()
    noise = sel_t["twin_noise_sd"]
    x = ["F1", "F2", "F3", "F4"]
    series = [(v.replace("M5v6_", ""), [m.loc[v, f] - m.loc["M4v6", f] for f in x], None, None, c, mk)
              for v, c, mk in zip(["M5v6_F1", "M5v6_F2", "M5v6_F3", "M5v6_F4", "M5v6_F5", "M5v6_F6"], bc.PALETTE, ["o", "s", "^", "D", "v", "P"]) if v in m.index]
    bc.lines(OUT / "rolling_folds.png", "Emotion gain on each rolling fold, per variant",
             f"AUC of each M5v6 variant minus M4v6 per fold (seed means). Dashed band: +/- the twin's seed noise SD ({noise:.4f}); gains inside it cannot be "
             f"told apart from noise. Rule 8 {'chose ' + sel_t['chosen'] if sel_t['passed_rule'] else 'found no eligible variant (fallback F5)'}.",
             [1, 2, 3, 4], series, "Fold (test day 11, 12, 13, 14 Nov)", "AUC gain over M4v6", fmt="{:+.4f}", zero=True, xticks=x,
             hline=noise, hline_label="+1 SD seed noise")
    lad = [("M4 (week 6)", "M4_lightgbm"), ("M4v6 (fuller features)", "M4v6"), ("M5v6 (+ emotion)", "M5v6")]
    vals = [float(per(mm, "dev")["auc"].mean()) for _, mm in lad]
    vv = [float(per(mm if mm != "M5v6" else m5src, "valid")["auc"].mean()) for _, mm in lad]
    bc.grouped(OUT / "feature_ladder.png", "From M4 to M4v6 to M5v6",
               "AUC on valid (fold F4) and on the final test (second look). M4v6 adds time-causal no-emotion features; M5v6 adds user-relative emotion.",
               [l for l, _ in lad], [("valid (F4)", vv, None, None, bc.BLUE), ("final test", vals, None, None, bc.ORANGE)], "AUC", fmt="{:.4f}",
               ylim=(min(vals + vv) - 0.01, max(vals + vv) + 0.008))
    curves = []
    for name in ("M6", "M7a", "M7b", "M7c", "M7ctrl"):
        for s in (42, 43, 44):
            p = RUN_DIR / f"{name}_s{s}" / "curve.csv"
            if p.exists():
                curves.append(pd.read_csv(p))
    cv = pd.concat(curves, ignore_index=True)
    cv.to_csv(TABLES / "v6_nrms_curves.csv", index=False)
    ep = sorted(cv["epoch"].unique())
    ser = []
    for i, (name, c) in enumerate((("M6", bc.BLUE), (sel_n["selected"], bc.ORANGE), ("M7ctrl", bc.GREY))):
        for s, mk in zip((42, 43, 44), ("o", "s", "^")):
            d = cv[(cv["model"] == name) & (cv["seed"] == s)].set_index("epoch")["valid_auc"].reindex(ep)
            ser.append((f"{name} seed {s}", d.tolist(), None, None, c, mk))
    bc.lines(OUT / "nrms_curves.png", "NRMS training: valid AUC per epoch",
             "Valid AUC after each epoch for M6, the selected M7 and its control, 3 seeds each (training stops one epoch after the best).",
             ep, ser, "Epoch", "AUC (valid)", fmt="{:.3f}", xticks=[str(e) for e in ep])
    ns = pd.read_csv(TABLES / "v6_nrms_seeds_valid.csv")
    agg = ns.groupby("model")["auc"].agg(["mean", "min", "max"]).reindex(["M6", "M7a", "M7b", "M7c", "M7ctrl"]).dropna()
    bc.bars(OUT / "nrms_bars.png", "NRMS with and without emotion",
            "Valid AUC of M6 (no emotion), the M7 variants, and M7ctrl (the selected M7 with emotion vectors permuted across items), mean of 3 seeds, bars = min to max.",
            agg.index.tolist(), agg["mean"].tolist(), "AUC (valid)", low=agg["min"].tolist(), high=agg["max"].tolist(), fmt="{:.4f}",
            colors=[bc.BLUE if i == "M6" else (bc.GREY if i == "M7ctrl" else bc.ORANGE) for i in agg.index], zero=False, base=0.6,
            lim=(0.6, float(agg["max"].max()) + 0.01))
    # regenerated headline comparison (old version archived first)
    arch = ARCHIVE_FIGS / "archive_v5"
    arch.mkdir(exist_ok=True)
    for f in [*BEHAVIOUR_FIGS.glob("*.png"), *(ARCHIVE_FIGS / "presentation").glob("*.png"), *(ARCHIVE_FIGS / "results").glob("*.png")]:
        dest = arch / f.parent.name / f.name
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            shutil.copy2(f, dest)
    order = [("M2_sbert", "M2"), ("M3_sbert_emotion", "M3"), ("M3b_plus", "M3b+"), ("M4_lightgbm", "M4"), ("M5_lightgbm_emotion", "M5"),
             ("M4v6", "M4v6"), ("M5v6", "M5v6"), ("M6_nrms", "M6\nNRMS"), (m7, "M7"), ("M7ctrl_nrms_permuted", "M7ctrl"), ("B0", "B0\nblend"), ("B1", "B1\nblend")]
    v, lo, hi = [], [], []
    for mm, _ in order:
        a = per(mm, "dev")["auc"].dropna().to_numpy()
        h = st.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
        v.append(a.mean()); lo.append(a.mean() - h); hi.append(a.mean() + h)
    emo = {"M3_sbert_emotion", "M3b_plus", "M5_lightgbm_emotion", "M5v6", m7, "B1"}
    bc.bars(BEHAVIOUR_FIGS / "e7_final_comparison.png", "Final test: every model, with and without emotion",
            f"Final-test AUC with 95% CIs (week-8 models are a second look at dev). Orange = with emotion, grey = control. Best: {best} {ladder[best]:.4f}.",
            [l for _, l in order], v, "AUC (final test)", low=lo, high=hi, fmt="{:.3f}",
            colors=[bc.GREY if mm == "M7ctrl_nrms_permuted" else (bc.ORANGE if mm in emo else bc.BLUE) for mm, _ in order], zero=False, base=0.5,
            lim=(0.5, max(hi) + 0.02))
    best6 = ["fair_twin.png", "rolling_folds.png", "nrms_bars.png", "feature_ladder.png", "audit_overlap.png", "twin_gain_by_consistency.png"]
    bc.contact_sheet([OUT / n for n in best6], OUT / "overview_v6.png", "Beyond Keywords, week 8: does emotion beat a fair twin?", cols=3)
    import subprocess, sys
    subprocess.run([sys.executable, str(RESULTS.parent / "scripts" / "06_12_overview.py")], check=False)


if __name__ == "__main__":
    main()
