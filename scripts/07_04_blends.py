"""Stage 5 (plan v6): blends of the tree and NRMS models. B0 = w z(M4v6) + (1 - w) z(M6), B1 = w z(M5v6) + (1 - w) z(M7), z-scores within
each impression, w chosen on valid (fold F4 predictions); B1b adds v z(personal match), kept only if it beats B1 on valid (p < 0.05).
Dev is evaluated once per blend from the already-saved dev predictions of its parts (no model is rerun).

Reads:  saved valid and dev scores of M4v6, M5v6 (its source on valid), M6 and the selected M7; frozen settings; behaviours (personal match).
Writes: results/tables/v6_blends_*.csv, valid and dev files of B0, B1 (and B1b if kept); results/figures/v6_improve/blend_weights.png.
Run:    python scripts/07_04_blends.py
"""
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import behaviour_charts as bc
from src.behaviour_runs import emotion_score, load_context
from src.config import FIGURES, FROZEN_DIR, RESULTS, TABLES
from src.evaluate import evaluate, save_results
from src.run_registry import append_run
from src.stats import paired_test

GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
V_GRID = [0.0, 0.05, 0.1, 0.2, 0.3]
NRMS_NAMES = {"M7a": "M7a_nrms_emotion_item", "M7b": "M7b_nrms_emotion_gated", "M7c": "M7c_nrms_emotion_both"}


def zscore(imp, x):
    s = pd.Series(np.asarray(x, dtype=np.float64))
    g = s.groupby(imp)
    sd = g.transform("std").fillna(0).to_numpy()
    return np.where(sd > 0, (s.to_numpy() - g.transform("mean").to_numpy()) / np.where(sd > 0, sd, 1.0), 0.0)


def scores(model, part, truth):
    s = pd.read_parquet(RESULTS / "scores" / f"{model}__{part}.parquet").set_index(["imp_key", "news_id"])["score"]
    return s.reindex(pd.MultiIndex.from_frame(truth[["imp_key", "news_id"]])).to_numpy()


def main():
    m5src = json.loads((FROZEN_DIR / "M5v6.json").read_text(encoding="utf-8"))["config"]["source"]
    m7 = NRMS_NAMES[json.loads((TABLES / "v6_nrms_selection.json").read_text(encoding="utf-8"))["selected"]]
    parts = {}
    for part in ("valid", "dev"):
        ctx = load_context(part, with_m2=False)
        imp = ctx.imp_codes
        z = {k: zscore(imp, scores(m, part, ctx.truth)) for k, m in
             (("tree0", "M4v6"), ("tree1", m5src if part == "valid" else "M5v6"), ("nrms0", "M6_nrms"), ("nrms1", m7))}
        z["emo"] = zscore(imp, emotion_score(ctx, 50, 1.0))
        parts[part] = (ctx, z)
    ctx, z = parts["valid"]
    curves, best = [], {}
    for name, a, b in (("B0", "tree0", "nrms0"), ("B1", "tree1", "nrms1")):
        res = []
        for w in GRID:
            per, s = evaluate(ctx.truth, ctx.truth[["imp_key", "news_id"]].assign(score=w * z[a] + (1 - w) * z[b]))
            res.append((w, float(s["auc"]), per))
            curves.append({"blend": name, "w_tree": w, "valid_auc": float(s["auc"])})
        best[name] = max(res, key=lambda r: (r[1], -abs(r[0] - 0.5)))
    w1 = best["B1"][0]
    vres = []
    for v in V_GRID:
        per, s = evaluate(ctx.truth, ctx.truth[["imp_key", "news_id"]].assign(score=w1 * z["tree1"] + (1 - w1) * z["nrms1"] + v * z["emo"]))
        vres.append((v, float(s["auc"]), per))
        curves.append({"blend": "B1b", "w_tree": w1, "v_emotion": v, "valid_auc": float(s["auc"])})
    v_best = max(vres, key=lambda r: (r[1], -r[0]))
    t = paired_test(best["B1"][2]["auc"], v_best[2]["auc"])
    keep_b1b = v_best[0] > 0 and t["mean_diff"] > 0 and t["p"] < 0.05
    pd.DataFrame(curves).to_csv(TABLES / "v6_blends_valid_curves.csv", index=False)
    choice = {"B0_w": best["B0"][0], "B1_w": w1, "B1b_v": v_best[0], "B1b_kept": bool(keep_b1b), "B1b_test": t, "m5v6_source": m5src, "m7": m7}
    (TABLES / "v6_blends_choice.json").write_text(json.dumps(choice, indent=2, default=float), encoding="utf-8")
    for part in ("valid", "dev"):
        c, zz = parts[part]
        blends = {"B0": best["B0"][0] * zz["tree0"] + (1 - best["B0"][0]) * zz["nrms0"], "B1": w1 * zz["tree1"] + (1 - w1) * zz["nrms1"]}
        if keep_b1b:
            blends["B1b"] = blends["B1"] + v_best[0] * zz["emo"]
        for name, sc in blends.items():
            out = c.truth[["imp_key", "news_id"]].assign(score=sc)
            per, s = evaluate(c.truth, out)
            save_results(name, part, out, per, s)
            append_run(name, part, "blend-of-saved-predictions" if part == "dev" else "blend-chosen-on-valid", choice, s)
            print(part, name, round(float(s["auc"]), 5), flush=True)
    cv = pd.DataFrame(curves)
    bc.lines(FIGURES / "v6_improve" / "blend_weights.png", "How much tree vs NRMS in the blend?",
             f"Valid AUC of the within-impression z-score blend by the tree weight w (0 = NRMS only, 1 = tree only). Chosen: B0 w = {best['B0'][0]}, "
             f"B1 w = {w1}. Chosen on valid, so optimistic.", GRID,
             [("B0: M4v6 + M6 (no emotion)", cv[cv["blend"] == "B0"]["valid_auc"].tolist(), None, None, bc.BLUE, "o"),
              ("B1: M5v6 + M7 (emotion)", cv[cv["blend"] == "B1"]["valid_auc"].tolist(), None, None, bc.ORANGE, "s")],
             "w (weight of the tree model)", "AUC (valid)", fmt="{:.4f}")


if __name__ == "__main__":
    main()
