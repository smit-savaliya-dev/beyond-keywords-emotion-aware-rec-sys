"""Week-9 multi-metric evaluation: one no-emotion model per approach (chosen on VALID AUC, standard setting) and its emotion twin.

No model is trained and no model is newly scored on dev: the inputs are the saved dev score files in results/scores/ (M7d's from its
one approved run, D107). The only rebuilt score vector is Combined + emotion at the Combined twin's own valid-chosen blend weight
(same recipe as scripts/08_03_final_metrics.py). Metrics come from src.evaluate.evaluate (src.metrics), and every per-impression
value is checked against the saved per-impression file of the main evaluation.

Gains: emotion model minus its twin on the same impressions. The gain is the difference of the two table means; its 95% CI and paired
t-test p are clustered by user (src.stats.clustered_paired_test); Holm over all 4 pairs x 8 metrics (standard setting). The
real-time pair (figure 10 only) is tested separately with its own Holm family of 8.

Reads:  results/tables/summary.csv (valid AUC of every version), final_all_metrics.csv (earlier same-weight blends), v7_blends_choice.json,
        results/scores/*__dev.parquet, results/per_impression/*__dev.parquet, data/processed/impressions.parquet (users).
Writes: results/tables/final_model_choice_valid.csv, final_metrics_all_models.csv, final_emotion_gains_all_metrics.csv,
        all_versions_full.csv.
Run:    python scripts/08_04_week9_metrics.py
"""
import importlib
import json
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import RESULTS, TABLES
from src.evaluate import evaluate, load_truth
from src.final_eval import load_users
from src.stats import clustered_paired_test, holm

STD, RT = "Standard", "Real-time"
METRICS = ["auc", "mrr", "ndcg@5", "ndcg@10", "p@5", "r@10", "hr@5", "hr@10"]
METRIC_NAMES = {"auc": "AUC", "mrr": "MRR", "ndcg@5": "nDCG@5", "ndcg@10": "nDCG@10", "p@5": "Precision@5", "r@10": "Recall@10",
                "hr@5": "Hit rate@5", "hr@10": "Hit rate@10"}
APPROACHES = ["Baselines", "Meaning", "LightGBM", "Neural", "Combined"]

# every version that has a summary row: code -> (approach, setting, emotion, role)
VERSIONS = {
    "M0_popularity": ("Baselines", STD, False, "final"), "M1_tfidf": ("Baselines", STD, False, "final"),
    "M2_sbert": ("Meaning", STD, False, "final"), "M3_sbert_emotion": ("Meaning", STD, True, "earlier version"),
    "M3b_plus": ("Meaning", STD, True, "final"),
    "M4_lightgbm": ("LightGBM", STD, False, "earlier version (week 4)"), "T1_M4cols": ("LightGBM", STD, False, "plan v7 comparison model"),
    "M4v6_R1": ("LightGBM", STD, False, "final"), "M5_lightgbm_emotion": ("LightGBM", STD, True, "earlier version (week 4)"),
    "M5_all_lightgbm": ("LightGBM", STD, True, "earlier variant"), "M5_item_only": ("LightGBM", STD, True, "earlier variant"),
    "M5_congruence_only": ("LightGBM", STD, True, "earlier variant"), "M5_permuted_items": ("LightGBM", STD, True, "control (shuffled emotion)"),
    "M5b_lightgbm_emotion": ("LightGBM", STD, True, "earlier version"), "M5v6_R1_F2": ("LightGBM", STD, True, "plan v7 candidate"),
    "M5v6_R1_F3": ("LightGBM", STD, True, "plan v7 candidate"), "M5v6_R1_F6": ("LightGBM", STD, True, "plan v7 candidate"),
    "M5v6_R1": ("LightGBM", STD, True, "final"),
    "M4v6": ("LightGBM", RT, False, "real-time setting"), "M5v6": ("LightGBM", RT, True, "real-time setting"),
    **{f"M5v6_F{i}": ("LightGBM", RT, True, "plan v6 candidate") for i in range(1, 7)},
    "M6_nrms": ("Neural", STD, False, "final"), "M7a_nrms_emotion_item": ("Neural", STD, True, "plan v6 candidate"),
    "M7b_nrms_emotion_gated": ("Neural", STD, True, "earlier version (plan v6)"), "M7c_nrms_emotion_both": ("Neural", STD, True, "plan v6 candidate"),
    "M7ctrl_nrms_permuted": ("Neural", STD, True, "control (shuffled emotion)"), "M7d_nrms_emotion_fusion": ("Neural", STD, True, "final"),
    "M7d_ctrl_permuted": ("Neural", STD, True, "control (shuffled emotion)"), "M7e_nrms_emotion_personal_head": ("Neural", STD, True, "plan v8 candidate"),
    "M7f_nrms_emotion_head_item": ("Neural", STD, True, "plan v8 candidate"),
    "B0_R1": ("Combined", STD, False, "final"), "B1_R1": ("Combined", STD, True, "own blend weight (not the fair twin)"),
    "B1_R1_same_weight": ("Combined", STD, True, "final"),
    "B0": ("Combined", RT, False, "real-time setting"), "B1": ("Combined", RT, True, "real-time setting"),
    "B1_same_weight": ("Combined", RT, True, "real-time setting"),
}
DISPLAY = {"M0_popularity": "Popularity", "M1_tfidf": "Keyword match (TF-IDF)", "M2_sbert": "Meaning match (Sentence-BERT)",
           "M3b_plus": "Meaning + emotion", "M4v6_R1": "LightGBM ranker", "M5v6_R1": "LightGBM + emotion", "M6_nrms": "Neural ranker (NRMS)",
           "M7d_nrms_emotion_fusion": "Neural + emotion", "B0_R1": "Combined ranker", "B1_R1_same_weight": "Combined + emotion",
           "M4v6": "LightGBM ranker", "M5v6": "LightGBM + emotion"}
# the emotion twin of each final no-emotion model (designated by the project's pre-declared rules: D95, D97, D106, plan v6 M3b+)
TWIN = {"M2_sbert": "M3b_plus", "M4v6_R1": "M5v6_R1", "M6_nrms": "M7d_nrms_emotion_fusion", "B0_R1": "B1_R1_same_weight"}
RT_PAIR = ("M4v6", "M5v6")


def valid_auc():
    s = pd.read_csv(TABLES / "summary.csv")
    return s[s["part"] == "valid"].set_index("model")["auc"]


def has_dev(code):
    return (RESULTS / "scores" / f"{code}__dev.parquet").exists() or code.endswith("_same_weight")


def choose():
    """Per approach: the standard-setting no-emotion model with the highest valid AUC among those that have an emotion twin and saved
    test-day predictions (a model without them cannot be shown without new training or a new test-day run)."""
    v = valid_auc()
    rows = []
    for code, (app, setting, emo, role) in VERSIONS.items():
        if setting != STD or emo or code not in v.index:
            continue
        rows.append({"approach": app, "model": code, "role": role, "valid_auc": float(v[code]), "has_emotion_twin": code in TWIN or app == "Baselines",
                     "has_test_predictions": has_dev(code)})
    c = pd.DataFrame(rows)
    c["eligible"] = c["has_emotion_twin"] & c["has_test_predictions"]
    c["rank_valid_auc"] = c.groupby("approach")["valid_auc"].rank(ascending=False, method="first").astype(int)
    c["chosen"] = False
    for app, g in c.groupby("approach"):
        if app == "Baselines":
            c.loc[g.index, "chosen"] = True                 # both baselines are shown
        else:
            c.loc[g[g["eligible"]]["valid_auc"].idxmax(), "chosen"] = True
    c["emotion_twin"] = c["model"].map(TWIN)
    tv = v.to_dict()
    tv["M5v6_R1"] = float(v["M5v6_R1_F3"])                   # frozen M5v6_R1 = candidate F3 (D95); its valid scores are F3's
    tv["B1_R1_same_weight"] = same_weight_valid_auc()
    c["twin_valid_auc"] = c["emotion_twin"].map(tv)
    c = c.sort_values(["approach", "rank_valid_auc"], key=lambda s: s.map(APPROACHES.index) if s.name == "approach" else s)
    c.to_csv(TABLES / "final_model_choice_valid.csv", index=False)
    return c


def same_weight_valid_auc():
    """Valid AUC of Combined + emotion at the Combined twin's weight, from saved valid scores (the recipe must reproduce B0_R1's)."""
    bl = importlib.import_module("07_04_blends")
    truth = load_truth("valid")
    imp = truth["imp_key"].to_numpy()
    w = float(json.loads((TABLES / "v7_blends_choice.json").read_text(encoding="utf-8"))["B0_R1_w"])

    def auc(tree, nrms):
        z = w * bl.zscore(imp, bl.scores(tree, "valid", truth)) + (1 - w) * bl.zscore(imp, bl.scores(nrms, "valid", truth))
        return float(evaluate(truth, truth[["imp_key", "news_id"]].assign(score=z))[1]["auc"])
    b0 = auc("M4v6_R1", "M6_nrms")
    assert abs(b0 - valid_auc()["B0_R1"]) < 5e-5, f"rebuilt B0_R1 valid AUC {b0} differs from the summary"
    return auc("M5v6_R1_F3", "M7b_nrms_emotion_gated")


def same_weight_blend(truth):
    mod = importlib.import_module("08_03_final_metrics")
    w = float(json.loads((TABLES / "v7_blends_choice.json").read_text(encoding="utf-8"))["B0_R1_w"])
    return mod.blend(truth, w, "M5v6_R1", "M7b_nrms_emotion_gated"), mod.blend(truth, w, "M4v6_R1", "M6_nrms"), w


def per_impression(code, truth):
    if code == "B1_R1_same_weight":
        scores, twin, w = same_weight_blend(truth)
        check, _ = evaluate(truth, twin)
        saved = pd.read_parquet(RESULTS / "per_impression" / "B0_R1__dev.parquet")
        assert np.allclose(check["auc"], saved["auc"], equal_nan=True), "rebuilt B0_R1 differs from the saved blend"
        source = f"same-weight blend (w_tree = {w}) of saved dev scores M5v6_R1 + M7b_nrms_emotion_gated"
    else:
        scores = pd.read_parquet(RESULTS / "scores" / f"{code}__dev.parquet")
        source = f"saved dev scores results/scores/{code}__dev.parquet"
    per, _ = evaluate(truth, scores)
    saved = RESULTS / "per_impression" / f"{code}__dev.parquet"
    if saved.exists():
        s = pd.read_parquet(saved)
        for k in METRICS:
            assert np.allclose(per[k], s[k], atol=1e-12, equal_nan=True), f"{code} {k} differs from the saved per-impression file"
    return per, source


def gains(pairs, per, users, family):
    rows = []
    for a, b in pairs:
        m = per[a].merge(per[b], on="imp_key", suffixes=("_a", "_b"), validate="one_to_one")
        g = m["imp_key"].map(users).to_numpy()
        assert not pd.isna(g).any(), "an impression has no user"
        for k in METRICS:
            t = clustered_paired_test(m[f"{k}_a"], m[f"{k}_b"], g)
            rows.append({"approach": VERSIONS[a][0], "setting": VERSIONS[a][1], "twin": a, "emotion_model": b, "twin_name": DISPLAY[a],
                         "emotion_name": DISPLAY[b], "metric": k, "metric_name": METRIC_NAMES[k], "impressions": t["n"], "users": t["clusters"],
                         "twin_value": t["mean_a"], "emotion_value": t["mean_b"], "gain": t["mean_diff"], "ci_low": t["ci_low"],
                         "ci_high": t["ci_high"], "p": t["p"], "dz": t["dz"], "holm_family": family})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy())
    out["significant_holm"] = out["p_holm"] < 0.05
    return out


def all_versions(choice, metrics):
    s = pd.read_csv(TABLES / "summary.csv")
    old = pd.read_csv(TABLES / "final_all_metrics.csv")
    old = old[old["model"].str.endswith("_same_weight")].assign(part="dev")
    s = pd.concat([s, old[["model", "part", "impressions"] + METRICS]], ignore_index=True)
    s = s.drop_duplicates(["model", "part"], keep="first")
    wide = s.pivot(index="model", columns="part", values=["auc", "ndcg@10", "mrr"])
    wide.columns = [f"{p}_{m}" for m, p in wide.columns]
    wide = wide.reset_index()
    info = pd.DataFrame([{"model": c, "approach": a, "setting": st, "emotion": e, "role": r} for c, (a, st, e, r) in VERSIONS.items()])
    out = info.merge(wide, on="model", how="outer", validate="one_to_one")
    assert out["approach"].notna().all(), f"versions without a description: {out[out['approach'].isna()]['model'].tolist()}"
    shown = set(metrics["model"])
    out["display_name_if_shown"] = out["model"].map(DISPLAY).where(out["model"].isin(shown))
    out["shown_in_figures"] = out["model"].isin(shown)
    out["chosen_on_valid"] = out["model"].isin(set(choice[choice["chosen"]]["model"]) | set(TWIN.values()))
    cols = ["model", "display_name_if_shown", "approach", "setting", "emotion", "role", "shown_in_figures", "chosen_on_valid",
            "valid_auc", "valid_ndcg@10", "valid_mrr", "dev_auc", "dev_ndcg@10", "dev_mrr"]
    out = out[cols].sort_values(["setting", "approach", "emotion", "model"], key=lambda c: c.map(APPROACHES.index) if c.name == "approach" else c)
    out.to_csv(TABLES / "all_versions_full.csv", index=False)
    return out


def main():
    t0 = time.time()
    choice = choose()
    print(choice[["approach", "model", "valid_auc", "eligible", "chosen", "emotion_twin", "twin_valid_auc"]].to_string(index=False))
    chosen = choice[choice["chosen"]].assign(order=lambda d: d["model"].map({"M0_popularity": 0, "M1_tfidf": 1}).fillna(2))
    shown = [c for app in APPROACHES for c in chosen[chosen["approach"] == app].sort_values("order")["model"].tolist()
             for c in ([c, TWIN[c]] if c in TWIN else [c])]
    truth = load_truth("dev")
    per, rows = {}, []
    for code in shown + list(RT_PAIR):
        per[code], source = per_impression(code, truth)
        app, setting, emo, _ = VERSIONS[code]
        rows.append({"model": code, "display_name": DISPLAY[code], "approach": app, "setting": setting, "emotion": emo,
                     **{k: float(per[code][k].mean()) for k in METRICS}, "impressions": len(per[code]), "source": source})
        print(f"{code:28s} auc {rows[-1]['auc']:.4f} ndcg@10 {rows[-1]['ndcg@10']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLES / "final_metrics_all_models.csv", index=False)
    users = load_users("dev")
    pairs = [(a, TWIN[a]) for a in shown if a in TWIN]
    g = pd.concat([gains(pairs, per, users, "standard: 4 pairs x 8 metrics"), gains([RT_PAIR], per, users, "real-time: 1 pair x 8 metrics")],
                  ignore_index=True)
    g.to_csv(TABLES / "final_emotion_gains_all_metrics.csv", index=False)
    all_versions(choice, metrics)
    print(g[g["metric"].isin(["auc", "ndcg@10"])][["approach", "setting", "metric", "gain", "ci_low", "ci_high", "p", "p_holm", "dz"]].to_string(index=False))
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
