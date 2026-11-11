"""Plan v7, Steps 1c-1f, 2, 3 and the v7 charts: R1 (previous days only, primary) vs R2 (same-day, secondary).

  --blends     R1 blends B0_R1 / B1_R1: weight chosen on valid (F4 predictions), dev from the saved dev files (no model is rerun)
  --report     fold evidence, survival of the M4v6 gain, dev families (Holm), shuffled-label checks, subgroups, error analysis, charts
Dev is read only from saved prediction files (a further look at dev, which was already seen in weeks 7-8). Nothing here chooses a model.
Reads: results/tables/v7_*.csv, v6_trees_folds.csv, data/processed/ltr_v6_eval_* and ltr_v7_eval_*, saved valid/dev scores and per-impression files.
Writes: results/tables/v7_*.csv, archive/figures/v7_final/*.png (+ a CSV per chart), docs/error_analysis_final.md (tables; summary added by hand).
"""
import argparse
import importlib
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import behaviour_charts as bc
from src.behaviour import assign_tertile, history_consistency, time_slot, window_profiles
from src.behaviour_runs import emotion_score, load_context
from src.config import ARCHIVE_FIGS, BEHAVIOUR_SPEC_VERSION, FIGURES, PROCESSED, RESULTS, ROOT, TABLES
from src.emotion_spec import load_spec
from src.evaluate import evaluate, save_results
from src.ltr_v6 import FOLDS, persona_model
from src.run_registry import append_run, load_rows
from src.stats import holm, paired_test
from src.subgroups import fast_metrics, group_gains, paired_family, shuffled_labels, user_bootstrap_ci
from src.where_it_helps import clicked_category, history_bin, top_categories

blends_v6 = importlib.import_module("07_04_blends")
OUT = ARCHIVE_FIGS / "v7_final"
M7, M7CTRL, M6 = "M7b_nrms_emotion_gated", "M7ctrl_nrms_permuted", "M6_nrms"
OLD_M4, OLD_M5 = "M4_lightgbm", "M5_lightgbm_emotion"


def sel():
    return json.loads((TABLES / "v7_selection.json").read_text(encoding="utf-8"))


def per(m, part):
    return pd.read_parquet(RESULTS / "per_impression" / f"{m}__{part}.parquet")


def chart_csv(name, frame):
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f"{name}.csv", index=False)


# ---------------------------------------------------------------- blends (R1)
def r1_blends():
    s = sel()
    parts = {}
    for part in ("valid", "dev"):
        ctx = load_context(part, with_m2=False)
        tree1 = s["chosen"] if part == "valid" else "M5v6_R1"
        z = {k: blends_v6.zscore(ctx.imp_codes, blends_v6.scores(m, part, ctx.truth)) for k, m in
             (("tree0", "M4v6_R1"), ("tree1", tree1), ("nrms0", M6), ("nrms1", M7))}
        parts[part] = (ctx, z)
    ctx, z = parts["valid"]
    curves, best = [], {}
    for name, a, b in (("B0_R1", "tree0", "nrms0"), ("B1_R1", "tree1", "nrms1")):
        res = []
        for w in blends_v6.GRID:
            _, sm = evaluate(ctx.truth, ctx.truth[["imp_key", "news_id"]].assign(score=w * z[a] + (1 - w) * z[b]))
            res.append((w, float(sm["auc"])))
            curves.append({"blend": name, "w_tree": w, "valid_auc": float(sm["auc"])})
        best[name] = max(res, key=lambda r: (r[1], -abs(r[0] - 0.5)))[0]
    pd.DataFrame(curves).to_csv(TABLES / "v7_blends_valid_curves.csv", index=False)
    (TABLES / "v7_blends_choice.json").write_text(json.dumps({"B0_R1_w": best["B0_R1"], "B1_R1_w": best["B1_R1"], "m7": M7}, indent=2), encoding="utf-8")
    rows = []
    for part in ("valid", "dev"):
        c, zz = parts[part]
        out = {"B0_R1": best["B0_R1"] * zz["tree0"] + (1 - best["B0_R1"]) * zz["nrms0"],
               "B1_R1": best["B1_R1"] * zz["tree1"] + (1 - best["B1_R1"]) * zz["nrms1"]}
        for name, sc in out.items():
            f = c.truth[["imp_key", "news_id"]].assign(score=sc)
            p, sm = evaluate(c.truth, f)
            save_results(name, part, f, p, sm)
            append_run(name, part, "blend-of-saved-predictions" if part == "dev" else "blend-chosen-on-valid", best | {"plan": "v7"}, sm)
            print(part, name, round(float(sm["auc"]), 5), flush=True)
        for w in sorted({best["B0_R1"], best["B1_R1"]}):          # declared same-weight check (plan v7)
            a = fast_metrics(c.truth["imp_key"].to_numpy(), c.truth["label"].to_numpy(), w * zz["tree0"] + (1 - w) * zz["nrms0"])
            b = fast_metrics(c.truth["imp_key"].to_numpy(), c.truth["label"].to_numpy(), w * zz["tree1"] + (1 - w) * zz["nrms1"])
            t = paired_test(a["auc"], b["auc"])
            rows.append({"part": part, "w_tree": w, "b1_minus_b0": t["mean_diff"], "ci_low": t["ci_low"], "ci_high": t["ci_high"], "p": t["p"], "dz": t["dz"]})
    pd.DataFrame(rows).to_csv(TABLES / "v7_blends_same_weight.csv", index=False)


# ---------------------------------------------------------------- folds
def fold_tables():
    v7 = pd.read_csv(TABLES / "v7_folds.csv")
    v6 = pd.read_csv(TABLES / "v6_trees_folds.csv")
    t = pd.concat([v6[v6["model"].isin(["M4v6", "M5v6_F2"])], v7], ignore_index=True)
    m = t.groupby(["model", "fold"])["auc"].mean().unstack()[list(FOLDS)]
    sd = t.groupby(["model", "fold"])["auc"].std().unstack()[list(FOLDS)]
    beh = pd.read_parquet(PROCESSED / "behaviors.parquet", columns=["imp_key", "user_id"])
    users = beh.set_index(beh["imp_key"].astype(str))["user_id"].astype(str)
    s = sel()
    pairs = {"R1": ("M4v6_R1", s["chosen"]), "R2": ("M4v6", "M5v6_F2"),
             "survival T3 - T1": ("T1_M4cols", "M4v6_R1"), "full gain T2 - T1": ("T1_M4cols", "M4v6"), "R2 - R1 twin": ("M4v6_R1", "M4v6")}
    rows, pooled = [], []
    for label, (a, b) in pairs.items():
        diffs, us = [], []
        for f in FOLDS:
            pa = pd.read_parquet(PROCESSED / (f"ltr_v6_eval_{a}_{f}.parquet" if a in ("M4v6", "M5v6_F2") else f"ltr_v7_eval_{a}_{f}.parquet"))
            pb = pd.read_parquet(PROCESSED / (f"ltr_v6_eval_{b}_{f}.parquet" if b in ("M4v6", "M5v6_F2") else f"ltr_v7_eval_{b}_{f}.parquet"))
            mm = pa[["imp_key", "auc"]].merge(pb[["imp_key", "auc"]], on="imp_key", suffixes=("_a", "_b"))
            tt = paired_test(mm["auc_a"], mm["auc_b"])
            d = (mm["auc_b"] - mm["auc_a"]).to_numpy()
            u = mm["imp_key"].astype(str).map(users).to_numpy()
            lo, hi = user_bootstrap_ci(d, u)
            rows.append({"comparison": label, "model_a": a, "model_b": b, "fold": f, "auc_a": m.loc[a, f], "auc_b": m.loc[b, f],
                         "gain_seed_mean": m.loc[b, f] - m.loc[a, f], "gain": tt["mean_diff"], "ci_low_user": lo, "ci_high_user": hi,
                         "p": tt["p"], "dz": tt["dz"], "seed_sd_a": sd.loc[a, f]})
            diffs.append(d)
            us.append(u)
        d, u = np.concatenate(diffs), np.concatenate(us)
        tt = paired_test(np.zeros_like(d), d)
        lo, hi = user_bootstrap_ci(d, u)
        r = pd.DataFrame(rows)
        r = r[r["comparison"] == label]
        pooled.append({"comparison": label, "model_a": a, "model_b": b, "mean_auc_a": float(m.loc[a].mean()), "mean_auc_b": float(m.loc[b].mean()),
                       "gain": tt["mean_diff"], "ci_low_user": lo, "ci_high_user": hi, "p": tt["p"], "dz": tt["dz"],
                       "folds_won_F2_F4": int((r.set_index("fold").loc[["F2", "F3", "F4"], "gain_seed_mean"] > 0).sum()),
                       "seed_noise_sd": float(sd.loc[a].mean())})
    folds, pooled = pd.DataFrame(rows), pd.DataFrame(pooled)
    tw = pooled["comparison"].isin(["R1", "R2"])
    pooled.loc[tw, "p_holm"] = holm(pooled.loc[tw, "p"].to_numpy())           # family: the two emotion-vs-twin comparisons
    full = m.loc["M4v6"] - m.loc["T1_M4cols"]
    surv = m.loc["M4v6_R1"] - m.loc["T1_M4cols"]
    survival = pd.DataFrame({"fold": list(FOLDS), "T1_old_M4_cols": m.loc["T1_M4cols"].to_numpy(), "T3_M4v6_R1": m.loc["M4v6_R1"].to_numpy(),
                             "T2_M4v6_R2": m.loc["M4v6"].to_numpy(), "gain_R2": full.to_numpy(), "gain_R1": surv.to_numpy(),
                             "share_surviving": (surv / full).to_numpy()})
    survival.loc[len(survival)] = ["mean", m.loc["T1_M4cols"].mean(), m.loc["M4v6_R1"].mean(), m.loc["M4v6"].mean(), full.mean(), surv.mean(),
                                   surv.mean() / full.mean()]
    folds.to_csv(TABLES / "v7_fold_evidence.csv", index=False)
    pooled.to_csv(TABLES / "v7_fold_pooled.csv", index=False)
    survival.to_csv(TABLES / "v7_survival_folds.csv", index=False)
    m.assign(mean=m.mean(axis=1), seed_sd=sd.mean(axis=1)).reset_index().to_csv(TABLES / "v7_fold_means.csv", index=False)
    return m, sd, folds, pooled, survival


# ---------------------------------------------------------------- dev families
def dev_families(ctx):
    users = ctx.users_by_imp
    primary = [("M5v6_R1 - M4v6_R1 (trees, R1)", "M4v6_R1", "M5v6_R1"), ("B1_R1 - B0_R1 (blends, R1)", "B0_R1", "B1_R1"),
               ("M7 - M6 (NRMS)", M6, M7), ("M7 - M7ctrl (emotion information)", M7CTRL, M7)]
    secondary = [("M4v6_R1 - M4 (survival)", OLD_M4, "M4v6_R1"), ("M4v6 (R2) - M4v6_R1 (same-day value)", "M4v6_R1", "M4v6"),
                 ("B0_R1 - M4v6_R1", "M4v6_R1", "B0_R1")]
    fp = paired_family([(n, per(a, "dev"), per(b, "dev")) for n, a, b in primary], users).assign(family="R1 primary (emotion vs twin)")
    fs = paired_family([(n, per(a, "dev"), per(b, "dev")) for n, a, b in secondary], users).assign(family="secondary")
    r2 = pd.read_csv(TABLES / "v6_final_dev_families.csv")
    r2 = r2[r2["family"].str.startswith("primary")].assign(family="R2 primary (plan v6, unchanged)")
    fam = pd.concat([fp, fs, r2], ignore_index=True)
    fam.to_csv(TABLES / "v7_dev_families.csv", index=False)
    a = {m: float(per(m, "dev")["auc"].mean()) for m in (OLD_M4, "M4v6", "M4v6_R1")}
    surv = {"dev_M4": a[OLD_M4], "dev_M4v6_R2": a["M4v6"], "dev_M4v6_R1": a["M4v6_R1"], "gain_R2": a["M4v6"] - a[OLD_M4],
            "gain_R1": a["M4v6_R1"] - a[OLD_M4], "share_surviving": (a["M4v6_R1"] - a[OLD_M4]) / (a["M4v6"] - a[OLD_M4])}
    pd.DataFrame([surv]).to_csv(TABLES / "v7_survival_dev.csv", index=False)
    sh = shuffled_labels(ctx.truth, 42)["label"].to_numpy()
    keys = pd.MultiIndex.from_frame(ctx.truth[["imp_key", "news_id"]])
    ctrl = []
    for mname in ("M4v6_R1", "M5v6_R1", "B0_R1", "B1_R1"):
        sc = pd.read_parquet(RESULTS / "scores" / f"{mname}__dev.parquet").set_index(["imp_key", "news_id"])["score"].reindex(keys).to_numpy()
        ctrl.append({"model": mname, "dev_auc_shuffled_labels": float(np.nanmean(fast_metrics(ctx.truth["imp_key"].to_numpy(), sh, sc)["auc"]))})
    pd.DataFrame(ctrl).to_csv(TABLES / "v7_dev_shuffled_controls.csv", index=False)
    return fam, surv, pd.DataFrame(ctrl)


# ---------------------------------------------------------------- subgroups (Step 2)
def splits_for(ctx, km, mu, sd, names, e1, top, news, spec):
    idx, r = ctx.imps["imp_key"].to_numpy(), ctx.imps["prof_row"].to_numpy()
    pair, _ = history_consistency(ctx.hist_rows, ctx.vectors)
    prof, used = window_profiles(ctx.hist_rows, ctx.vectors, 50, 0.9)
    lab = np.where(used >= 5, km.predict((prof - mu) / sd), -1)
    return {"consistency tertile": pd.Series(assign_tertile(pair[r], e1["cuts_fit"]), index=idx),
            "persona": pd.Series([names[i] if i >= 0 else "too little history" for i in lab[r]], index=idx),
            "clicked category": clicked_category(ctx.truth, news, top),
            "known history items": pd.Series(history_bin(used[r]), index=idx),
            "time slot": pd.Series(time_slot(ctx.imps["time"], spec["time_slots"]), index=idx)}, pair


def subgroups(ctx_dev):
    spec = load_spec(BEHAVIOUR_SPEC_VERSION)["behaviour"]
    e1 = json.loads((TABLES / "behaviour_e1_meta.json").read_text(encoding="utf-8"))
    k = json.loads((TABLES / "behaviour_e8_meta.json").read_text(encoding="utf-8"))
    fit = load_context("fit", with_m2=False)
    km, mu, sd = persona_model(fit, k=k["k"])
    del fit
    news = pd.read_parquet(PROCESSED / "news.parquet", columns=["news_id", "category"]).set_index("news_id")["category"].astype(str)
    fc = load_rows(PROCESSED, ["fit"], ["news_id", "label"])
    top = top_categories(fc.loc[fc["label"] == 1, "news_id"].astype(str).to_numpy(), news)
    rows = []
    s = sel()
    for part in ("valid", "dev"):
        ctx = ctx_dev if part == "dev" else load_context("valid", with_m2=False)
        sp, _ = splits_for(ctx, km, mu, sd, k["names"], e1, top, news, spec)
        r1 = ("M4v6_R1", s["chosen"] if part == "valid" else "M5v6_R1")
        r2 = ("M4v6", "M5v6_F2" if part == "valid" else "M5v6")
        dd = {}
        for regime, (a, b) in (("R1", r1), ("R2", r2)):
            pa, pb = per(a, part), per(b, part)
            dd[regime] = pb.set_index("imp_key")["auc"] - pa.set_index("imp_key")["auc"]
            for split, groups in sp.items():
                rows.append(group_gains(pa, pb, groups, ctx.users_by_imp, metrics=("auc",), name_a=a, name_b=b).assign(part=part, regime=regime, split=split))
        diff = (dd["R1"] - dd["R2"].reindex(dd["R1"].index))        # R1 gain minus R2 gain, per impression
        for split, groups in sp.items():
            g = groups.reindex(diff.index)
            for name, x in diff.groupby(g):
                t = paired_test(np.zeros(len(x)), x.to_numpy())
                rows.append(pd.DataFrame([{"group": name, "metric": "auc", "model_a": "R2 gain", "model_b": "R1 gain", "impressions": t["n"],
                                           "gain": t["mean_diff"], "ci_low": t["ci_low"], "ci_high": t["ci_high"], "dz": t["dz"], "p": t["p"],
                                           "part": part, "regime": "R1 minus R2", "split": split}]))
    t = pd.concat(rows, ignore_index=True)
    for part in ("valid", "dev"):
        for regime in ("R1", "R2", "R1 minus R2"):
            mask = (t["part"] == part) & (t["regime"] == regime) & t["p"].notna()
            t.loc[mask, "p_holm"] = holm(t.loc[mask, "p"].to_numpy())     # Holm over all declared cells of the regime
    t.to_csv(TABLES / "v7_subgroups.csv", index=False)
    return t, k["names"]


# ---------------------------------------------------------------- error analysis (Step 3)
def error_analysis(ctx):
    a, b = per("M4v6_R1", "dev"), per("M5v6_R1", "dev")
    d = a[["imp_key", "auc"]].merge(b[["imp_key", "auc"]], on="imp_key", suffixes=("_twin", "_emotion"))
    d["gain"] = d["auc_emotion"] - d["auc_twin"]
    size = ctx.truth.groupby("imp_key", sort=False).size()
    d = d[d["imp_key"].map(size) >= 10].dropna(subset=["gain"])
    pick = pd.concat([d.nlargest(10, "gain").assign(kind="helped"), d.nsmallest(10, "gain").assign(kind="hurt")], ignore_index=True)
    news = pd.read_parquet(PROCESSED / "news.parquet", columns=["news_id", "category", "title"]).set_index("news_id")
    keys = pd.MultiIndex.from_frame(ctx.truth[["imp_key", "news_id"]])
    st = {m: pd.read_parquet(RESULTS / "scores" / f"{m}__dev.parquet").set_index(["imp_key", "news_id"])["score"].reindex(keys).to_numpy()
          for m in ("M4v6_R1", "M5v6_R1")}
    pm = emotion_score(ctx, 50, 0.9)
    pair, _ = history_consistency(ctx.hist_rows, ctx.vectors)
    from src.config import GO_PCT_COLUMNS
    emo_names = [c.replace("go_pct_", "") for c in GO_PCT_COLUMNS]
    t = ctx.truth.assign(s_twin=st["M4v6_R1"], s_emo=st["M5v6_R1"], pm=pm, row=np.arange(len(ctx.truth)))
    t["rank_twin"] = t.groupby("imp_key")["s_twin"].rank(ascending=False, method="first")
    t["rank_emo"] = t.groupby("imp_key")["s_emo"].rank(ascending=False, method="first")
    t["pm_rank"] = t.groupby("imp_key")["pm"].rank(ascending=False, method="first")
    prof_row = ctx.imps.set_index("imp_key")["prof_row"]
    used = window_profiles(ctx.hist_rows, ctx.vectors, 50, 0.9)[1]
    out = []
    for _, r in pick.iterrows():
        c = t[(t["imp_key"] == r["imp_key"]) & (t["label"] == 1)].iloc[0]
        v = ctx.vectors[ctx.item_idx[int(c["row"])]]
        topk = np.argsort(-v)[:2]
        pr = int(prof_row[r["imp_key"]])
        out.append({"kind": r["kind"], "imp_key": r["imp_key"], "candidates": int(size[r["imp_key"]]),
                    "clicks": int(t[(t["imp_key"] == r["imp_key"])]["label"].sum()), "auc_twin": r["auc_twin"], "auc_emotion": r["auc_emotion"],
                    "gain": r["gain"], "headline": str(news.loc[c["news_id"], "title"]), "category": str(news.loc[c["news_id"], "category"]),
                    "top_emotions": ", ".join(f"{emo_names[j]} {v[j]:.2f}" for j in topk), "personal_match": float(c["pm"]),
                    "personal_match_rank": int(c["pm_rank"]), "rank_twin": int(c["rank_twin"]), "rank_emotion": int(c["rank_emo"]),
                    "consistency": float(pair[pr]), "history_items": int(used[pr])})
    e = pd.DataFrame(out)
    e.to_csv(TABLES / "v7_error_analysis.csv", index=False)
    allg = d["gain"].to_numpy()
    return e, allg


def write_error_md(e, allg):
    def table(k):
        x = e[e["kind"] == k]
        lines = ["| # | Clicked headline | Category | Top emotions | Personal match (rank in list) | Consistency | History | Rank twin -> emotion | AUC gain |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for i, (_, r) in enumerate(x.iterrows(), 1):
            cons = "n/a" if not np.isfinite(r["consistency"]) else f"{r['consistency']:.2f}"
            lines.append(f"| {i} | {r['headline'].replace('|', '/')} | {r['category']} | {r['top_emotions']} | {r['personal_match']:+.3f} ({r['personal_match_rank']} of "
                         f"{r['candidates']}) | {cons} | {r['history_items']} | {r['rank_twin']} -> {r['rank_emotion']} | {r['gain']:+.3f} |")
        return "\n".join(lines)
    text = f"""# Error analysis: R1 emotion model (M5v6_R1) against its twin (M4v6_R1)

Final test (dev), read from the saved prediction files (a further look at dev). Impressions with at least 10 candidates
({len(allg):,} impressions). Per impression: AUC of the emotion model minus AUC of the twin. Median gain {np.median(allg):+.4f};
{(allg > 0).mean():.1%} of impressions improve, {(allg < 0).mean():.1%} get worse, {(allg == 0).mean():.1%} unchanged.
Columns: the first clicked headline, its two strongest emotions (GoEmotions percentile among all articles, 0-1), its personal match (how much closer it is to the user's
emotional history than to the average user's; rank 1 = best match in the list), the user's consistency (how similar their history items are
emotionally, 0-1), known history items, and where each model put the clicked story (1 = top).

## Summary
SUMMARY_PLACEHOLDER

## 10 impressions where emotion helped most
{table("helped")}

## 10 impressions where emotion hurt most
{table("hurt")}

Chart: `archive/figures/v7_final/error_analysis.png`. Table: `results/tables/v7_error_analysis.csv`.
"""
    path = ROOT / "docs" / "error_analysis_final.md"
    if path.exists() and "SUMMARY_PLACEHOLDER" not in path.read_text(encoding="utf-8"):
        old = path.read_text(encoding="utf-8")
        summary = old.split("## Summary\n", 1)[1].split("\n## ", 1)[0].strip()
        text = text.replace("SUMMARY_PLACEHOLDER", summary)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- charts
def charts(m, sd, folds, pooled, survival, fam, surv_dev, sub, names, e, allg):
    OUT.mkdir(parents=True, exist_ok=True)
    s = sel()
    # 1. regime chart: dev AUC, without / with emotion, three settings
    lad = [("Old M4\n(week 6)", OLD_M4, OLD_M5), ("M4v6 R2\n(same-day, real-time)", "M4v6", "M5v6"), ("M4v6 R1\n(previous days, standard)", "M4v6_R1", "M5v6_R1")]
    rows = []
    for lab, a, b in lad:
        for kind, mname in (("without emotion", a), ("with emotion", b)):
            x = per(mname, "dev")["auc"].dropna().to_numpy()
            h = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
            rows.append({"setting": lab.replace("\n", " "), "series": kind, "model": mname, "dev_auc": x.mean(), "ci_low": x.mean() - h, "ci_high": x.mean() + h})
    r = pd.DataFrame(rows)
    chart_csv("regime_chart", r)
    w, wo = r[r["series"] == "with emotion"], r[r["series"] == "without emotion"]
    vals = r["dev_auc"].to_numpy()
    bc.grouped(OUT / "regime_chart.png", "M4v6's jump needs same-day clicks",
               f"Final-test AUC (a further look at dev) with 95% CIs. R2 uses earlier clicks of the same day; R1, the standard setting, only previous days. "
               f"M4v6 is +{surv_dev['gain_R2']:.3f} over the old M4 in R2 but {surv_dev['gain_R1']:+.3f} in R1; emotion adds +0.001 in R2 and +0.005 in R1.",
               [l for l, _, _ in lad], [("without emotion", wo["dev_auc"].tolist(), wo["ci_low"].tolist(), wo["ci_high"].tolist(), bc.BLUE),
                                        ("with emotion", w["dev_auc"].tolist(), w["ci_low"].tolist(), w["ci_high"].tolist(), bc.ORANGE)],
               "AUC (final test)", fmt="{:.4f}", ylim=(float(vals.min()) - 0.01, float(vals.max()) + 0.008), legend_loc="upper left")
    # 2. per-fold emotion gain with the seed-noise band
    x = list(FOLDS)
    ser, rows = [], []
    for v, c, mk in zip(["M5v6_R1_F2", "M5v6_R1_F3", "M5v6_R1_F6"], [bc.ORANGE, bc.PINK, bc.YELLOW], ["s", "^", "P"]):
        g = [m.loc[v, f] - m.loc["M4v6_R1", f] for f in x]
        ser.append((f"R1 {v.split('_')[-1]}" + (" (chosen)" if v == s["chosen"] else ""), g, None, None, c, mk))
        rows += [{"series": v, "regime": "R1", "fold": f, "gain": gg} for f, gg in zip(x, g)]
    g2 = [m.loc["M5v6_F2", f] - m.loc["M4v6", f] for f in x]
    ser.append(("R2 F2 (M5v6)", g2, None, None, bc.BLUE, "o"))
    rows += [{"series": "M5v6_F2", "regime": "R2", "fold": f, "gain": gg} for f, gg in zip(x, g2)]
    noise1, noise2 = float(sd.loc["M4v6_R1"].mean()), float(sd.loc["M4v6"].mean())
    chart_csv("fold_gains", pd.DataFrame(rows).assign(noise_sd_R1=noise1, noise_sd_R2=noise2))
    bc.lines(OUT / "fold_gains.png", "R1: emotion beats its twin on every fold; R2: near the noise",
             f"AUC of each emotion variant minus its no-emotion twin per fold (seed means), R1 and R2. Dashed: +1 SD of the R1 twin's seed noise "
             f"({noise1:.4f}; R2 {noise2:.4f}). Rule 8 in R1: {'chose ' + s['chosen'] if s['passed_rule'] else 'no variant passed (declared F2 used)'}.",
             [1, 2, 3, 4], ser, "Fold (test day 11, 12, 13, 14 Nov)", "AUC gain over the twin", fmt="{:+.4f}", zero=True, xticks=x,
             hline=noise1, hline_label="+1 SD seed noise (R1)")
    # 3. survival per fold
    sv = survival[survival["fold"] != "mean"]
    chart_csv("survival_folds", survival)
    bc.grouped(OUT / "survival_folds.png", "Without same-day clicks, almost none of M4v6's gain remains",
               f"Fold AUC (seed means): old M4 columns (T1), M4v6 in R1 (previous days only) and M4v6 in R2 (same-day). R1 keeps "
               f"{survival.iloc[-1]['share_surviving']:.0%} of the fold gain (+{survival.iloc[-1]['gain_R1']:.4f} of +{survival.iloc[-1]['gain_R2']:.4f}); "
               f"on the final test it is {surv_dev['gain_R1']:+.4f} below the old M4.",
               x, [("old M4 columns", sv["T1_old_M4_cols"].tolist(), None, None, bc.GREY), ("M4v6 R1", sv["T3_M4v6_R1"].tolist(), None, None, bc.BLUE),
                   ("M4v6 R2", sv["T2_M4v6_R2"].tolist(), None, None, bc.SKY)], "AUC (fold test day)", fmt="{:.3f}",
               ylim=(float(sv[["T1_old_M4_cols", "T3_M4v6_R1", "T2_M4v6_R2"]].min().min()) - 0.01, float(sv["T2_M4v6_R2"].max()) + 0.01), legend_loc="upper right")
    # 4. emotion vs twin, both regimes, folds and dev
    fp = fam.set_index("comparison")
    p1 = pooled.set_index("comparison")
    labs = ["Trees R1\nM5v6_R1 - M4v6_R1", "Trees R2\nM5v6 - M4v6", "Blends R1\nB1_R1 - B0_R1", "Blends R2\nB1 - B0"]
    dev_keys = ["M5v6_R1 - M4v6_R1 (trees, R1)", "M5v6 - M4v6 (trees)", "B1_R1 - B0_R1 (blends, R1)", "B1 - B0 (blends)"]
    fold_v = [p1.loc["R1", "gain"], p1.loc["R2", "gain"], np.nan, np.nan]
    fold_lo = [p1.loc["R1", "ci_low_user"], p1.loc["R2", "ci_low_user"], np.nan, np.nan]
    fold_hi = [p1.loc["R1", "ci_high_user"], p1.loc["R2", "ci_high_user"], np.nan, np.nan]
    dv = [fp.loc[k, "mean_diff"] for k in dev_keys]
    dlo = [fp.loc[k, "user_ci_low"] for k in dev_keys]
    dhi = [fp.loc[k, "user_ci_high"] for k in dev_keys]
    chart_csv("twin_both_regimes", pd.DataFrame({"comparison": dev_keys, "folds_gain": fold_v, "folds_ci_low": fold_lo, "folds_ci_high": fold_hi,
                                                 "dev_gain": dv, "dev_ci_low_user": dlo, "dev_ci_high_user": dhi,
                                                 "dev_p_holm": [fp.loc[k, "p_holm"] for k in dev_keys]}))
    bc.grouped(OUT / "twin_both_regimes.png", "Emotion against the same model without it, in both settings",
               "AUC gain over the no-emotion twin: rolling folds F1-F4 pooled (primary; trees only) and the final test (further look at dev), "
               "95% user-level CIs; Holm p within each regime's family. Blend gains mix in different blend weights (see same-weight check).",
               labs, [("folds F1-F4 (primary)", fold_v, fold_lo, fold_hi, bc.BLUE), ("final test", dv, dlo, dhi, bc.ORANGE)],
               "AUC gain over the twin", fmt="{:+.4f}", zero=True, notes=[f"Holm p {fp.loc[k, 'p_holm']:.2g}" for k in dev_keys])
    # 5. subgroups in R1 (consistency tertile + persona), R1 and R2 side by side
    d = sub[(sub["part"] == "dev") & (sub["regime"].isin(["R1", "R2"])) & (sub["split"].isin(["consistency tertile", "persona"]))]
    groups = ["low", "mid", "high", "no_history", *names, "too little history"]
    groups = [g for g in groups if g in set(d["group"])]
    xx = d.set_index(["regime", "group"])
    ser = []
    for reg, c in (("R1", bc.ORANGE), ("R2", bc.BLUE)):
        ser.append((f"{reg}: M5v6{'_R1' if reg == 'R1' else ''} - twin", [xx.loc[(reg, g), "gain"] for g in groups],
                    [xx.loc[(reg, g), "ci_low"] for g in groups], [xx.loc[(reg, g), "ci_high"] for g in groups], c))
    chart_csv("subgroups_r1", d)
    bc.grouped(OUT / "subgroups_r1.png", "Where emotion helps: consistency and reader persona",
               "Final-test AUC gain of the emotion model over its twin per declared group (exploratory), 95% user-bootstrap CIs; "
               "Holm over all declared cells in v7_subgroups.csv. Consistency: week-7 fit tertiles. " + ", ".join(names) + ".",
               [g.split(": ")[0] if ": " in g else g.replace("too little history", "little\nhistory").replace("no_history", "no\nhistory") for g in groups],
               ser, "AUC gain over the twin (final test)", fmt="{:+.4f}", zero=True)
    # 6. error analysis
    ee = e.copy()
    ee["label"] = [f"{h[:42]}{'...' if len(h) > 42 else ''} ({c}), rank {a} -> {b}"
                   for h, c, a, b in zip(ee["headline"], ee["category"], ee["rank_twin"], ee["rank_emotion"])]
    ee = ee.sort_values("gain")
    chart_csv("error_analysis", ee[["kind", "imp_key", "headline", "category", "gain", "rank_twin", "rank_emotion", "personal_match", "consistency"]])
    fig, ax = bc.new("Emotion's biggest wins and losses", grid="x", size=(14, 9))
    ax.barh(range(len(ee)), ee["gain"], color=[bc.ORANGE if g > 0 else bc.BLUE for g in ee["gain"]], height=0.7)
    ax.set_yticks(range(len(ee)), ee["label"], fontsize=11)
    for y, g in enumerate(ee["gain"]):
        ax.text(g + (0.02 if g > 0 else -0.02), y, f"{g:+.2f}", va="center", ha="left" if g > 0 else "right", fontsize=11, fontweight="bold")
    ax.axvline(0, color=bc.INK, linewidth=1.5)
    lim = float(np.abs(ee["gain"]).max()) * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("AUC gain on one impression (emotion model minus twin, final test)")
    fig.subplots_adjust(left=0.5, right=0.97, top=0.92, bottom=0.15)
    bc.save(fig, OUT / "error_analysis.png", f"R1, impressions with 10+ candidates; label = clicked headline, category, rank of the clicked story twin -> emotion. "
            f"Across all {len(allg):,} such impressions the mean gain is {np.mean(allg):+.4f}, the median {np.median(allg):+.4f}: extremes mostly cancel.")
    best6 = ["regime_chart.png", "survival_folds.png", "twin_both_regimes.png", "fold_gains.png", "subgroups_r1.png", "error_analysis.png"]
    bc.contact_sheet([OUT / n for n in best6], OUT / "overview_v7.png", "Beyond Keywords, final check: same-day clicks, regimes and emotion", cols=3)


def single_feature_auc():
    """Diagnostic: mean per-impression AUC of each behavioural column alone on valid (fold F4 test day), same-day vs previous-days-only."""
    regimes = importlib.import_module("08_01_regimes")
    pairs = [("pop_ctr", "pop_ctr")] + [(c, f"pd_{c}") for c in ("ctr_1h", "ctr_6h", "ctr_24h", "shows_24h", "exposure_log", "age_hours",
                                                                  "user_log_clicks", "user_log_cat_share")]
    cols = list(dict.fromkeys([c for p in pairs for c in p]))
    _, te = regimes.load_rows(cols, test=FOLDS["F4"])
    k, y = te["imp_key"].to_numpy(), te["label"].to_numpy()

    def auc(c):
        x = te[c].to_numpy(np.float64)
        return float(np.nanmean(fast_metrics(k, y, np.where(np.isnan(x), -1, x))["auc"]))
    t = pd.DataFrame([{"feature": a, "same_day_auc_R2": auc(a), "previous_days_auc_R1": auc(b)} for a, b in pairs])
    t.to_csv(TABLES / "v7_single_feature_auc.csv", index=False)
    return t


def seen_vs_new(ctx_dev):
    """POST-HOC EXPLORATORY (added after the R1 fold results, not used for the headline): emotion gain over the twin among candidates already
    shown before their day (pd_exposure_log > 0, label-free) vs candidates new that day, per impression (restricted AUC), valid and dev."""
    from src.subgroups import restricted_metrics
    pd_all = pd.read_parquet(PROCESSED / "ltr_v7_pd_all.parquet", columns=["imp_key", "news_id", "pd_exposure_log"]).set_index(["imp_key", "news_id"])
    s = sel()
    rows = []
    for part in ("valid", "dev"):
        ctx = ctx_dev if part == "dev" else load_context("valid", with_m2=False)
        keys = pd.MultiIndex.from_frame(ctx.truth[["imp_key", "news_id"]].astype(str))
        seen = pd_all["pd_exposure_log"].reindex(keys).to_numpy() > 0
        pairs = {"R1": ("M4v6_R1", s["chosen"] if part == "valid" else "M5v6_R1"), "R2": ("M4v6", "M5v6_F2" if part == "valid" else "M5v6"),
                 "R1 twin vs old M4": (OLD_M4, "M4v6_R1"), "R2 twin vs old M4": (OLD_M4, "M4v6")}
        for regime, (a, b) in pairs.items():
            sa, sb = (blends_v6.scores(mm, part, ctx.truth) for mm in (a, b))
            for name, mask in (("seen before the day", seen), ("new that day", ~seen)):
                pa, pb = restricted_metrics(ctx.truth, sa, mask), restricted_metrics(ctx.truth, sb, mask)
                mm = pa[["imp_key", "auc"]].merge(pb[["imp_key", "auc"]], on="imp_key", suffixes=("_a", "_b")).dropna()
                t = paired_test(mm["auc_a"], mm["auc_b"])
                rows.append({"part": part, "regime": regime, "items": name, "candidate_share": float(mask.mean()), "impressions": t["n"],
                             "auc_twin": t["mean_a"], "auc_emotion": t["mean_b"], "gain": t["mean_diff"], "ci_low": t["ci_low"], "ci_high": t["ci_high"],
                             "p": t["p"], "dz": t["dz"], "label": "POST-HOC EXPLORATORY"})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "v7_posthoc_seen_vs_new.csv", index=False)
    return t


def report():
    single_feature_auc()
    m, sd, folds, pooled, survival = fold_tables()
    ctx = load_context("dev", with_m2=False)
    fam, surv_dev, ctrl = dev_families(ctx)
    seen_vs_new(ctx)
    sub, names = subgroups(ctx)
    e, allg = error_analysis(ctx)
    write_error_md(e, allg)
    charts(m, sd, folds, pooled, survival, fam, surv_dev, sub, names, e, allg)
    pd.set_option("display.width", 250)
    print(m.assign(mean=m.mean(axis=1), seed_sd=sd.mean(axis=1)).round(5).to_string(), survival.round(4).to_string(index=False),
          pooled.round(5).to_string(index=False), folds.round(5).to_string(index=False), fam.round(5).to_string(index=False),
          pd.DataFrame([surv_dev]).round(4).to_string(index=False), ctrl.round(4).to_string(index=False), sep="\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blends", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.blends:
        r1_blends()
    if a.report:
        report()


if __name__ == "__main__":
    main()
