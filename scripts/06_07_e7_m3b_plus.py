"""E7a (plan v5): M3b+ = M2 + emotion with the best pre-declared combination of the E1 and E2 findings, chosen on valid in a fixed order:
(1) the E2 history window, (2) the same-day blend if the E2 value rule kept it, (3) alpha per consistency tertile if it beats alpha 0.8 on
valid (p < 0.05), else 0.8. Frozen, then dev once.

Reads:  valid (and, for --final-dev, dev) behaviours and impressions; saved M2 scores; behaviour_e1_meta.json, behaviour_e2_choice.json; spec v5.
Writes: --tune: results/frozen/M3b_plus.json, valid scores and per-impression files, behaviour_e7_m3bp_valid.csv and _meta.json.
        --final-dev: dev files, a summary.csv row, a runs_log.csv row, behaviour_e7_m3bp_dev_meta.json (same-day coverage on dev).
Run:    python scripts/06_07_e7_m3b_plus.py --tune
        python scripts/06_07_e7_m3b_plus.py --final-dev    # once, after the frozen marker is in docs/decisions.md
"""
import argparse
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.behaviour import assign_tertile, history_consistency
from src.behaviour_runs import emotion_score, fused_scores, load_context, metrics, same_day
from src.config import BEHAVIOUR_SPEC_VERSION, M2_NAME, M3BP_NAME, PROCESSED, SEED, TABLES
from src.emotion_spec import load_spec
from src.evaluate import evaluate, save_results
from src.provenance import run_info
from src.run_registry import append_run, check_final_dev, frozen_marker, load_frozen, save_frozen
from src.stats import paired_test
from src.subgroups import paired_family, shuffled_labels

TERTILES = ["low", "mid", "high"]


def build(ctx, cfg):
    """Fused M3b+ scores of a part for a frozen config, plus the same-day coverage."""
    sd, used, _ = same_day(ctx) if cfg["blend_a"] < 1.0 else (None, np.zeros(len(ctx.imps), dtype=np.int64), None)
    emo = emotion_score(ctx, cfg["window"]["size"], cfg["window"]["decay"], sd, cfg["blend_a"])
    pair, _ = history_consistency(ctx.hist_rows, ctx.vectors)
    tert = assign_tertile(pair[ctx.imps["prof_row"].to_numpy()], cfg["consistency_cuts"])
    alpha = np.array([cfg["alpha_by_tertile"].get(t, cfg["alpha_default"]) for t in tert])
    return fused_scores(ctx, emo, None, alpha), emo, tert, used


def tune():
    spec = load_spec(BEHAVIOUR_SPEC_VERSION)["behaviour"]
    e1 = json.loads((TABLES / "behaviour_e1_meta.json").read_text(encoding="utf-8"))
    e2 = json.loads((TABLES / "behaviour_e2_choice.json").read_text(encoding="utf-8"))
    ctx = load_context("valid")
    users = ctx.users_by_imp
    base_cfg = {"window": e2["window"], "blend_a": e2["blend_a"], "consistency_cuts": e1["cuts_fit"], "alpha_default": 0.8,
                "alpha_by_tertile": {}, "m2_config_hash": load_frozen(M2_NAME)["config_hash"]}
    uniform, emo, tert, _ = build(ctx, base_cfg)
    per_uniform = metrics(ctx, uniform)
    rows, best = [], {}
    for a in spec["adaptive_alpha_grid"]:
        per = metrics(ctx, fused_scores(ctx, emo, a))
        for t in TERTILES:
            rows.append({"alpha": a, "tertile": t, "valid_auc": float(np.nanmean(per["auc"].to_numpy()[tert == t]))})
    grid = pd.DataFrame(rows)
    for t in TERTILES:
        g = grid[grid["tertile"] == t]
        top = g[g["valid_auc"] == g["valid_auc"].max()]
        best[t] = float(top["alpha"].iloc[np.argmin(np.abs(top["alpha"].to_numpy() - 0.8))])
    adaptive = metrics(ctx, fused_scores(ctx, emo, None, np.array([best.get(t, 0.8) for t in tert])))
    test = paired_test(per_uniform["auc"], adaptive["auc"])
    keep = test["mean_diff"] > 0 and test["p"] < 0.05
    cfg = {**base_cfg, "alpha_by_tertile": best if keep else {}, "adaptive_kept": bool(keep)}
    scores, _, _, _ = build(ctx, cfg)
    truth = ctx.truth
    out = truth[["imp_key", "news_id"]].assign(score=scores)
    per, summary = evaluate(truth, out)
    m2 = metrics(ctx, ctx.m2)
    sh = shuffled_labels(truth, SEED)["label"].to_numpy()
    fam = paired_family([("M3b+ - M2", m2, metrics(ctx, scores)), ("M3b+ - uniform alpha", per_uniform, metrics(ctx, scores))], users)
    ctrl = paired_family([("M3b+ - M2, labels shuffled", metrics(ctx, ctx.m2, sh), metrics(ctx, scores, sh))], users).assign(family="control")
    table = pd.concat([fam, ctrl], ignore_index=True)
    grid.to_csv(TABLES / "behaviour_e7_m3bp_alpha_grid.csv", index=False)
    table.to_csv(TABLES / "behaviour_e7_m3bp_valid.csv", index=False)
    save_results(M3BP_NAME, "valid", out, per, summary)
    append_run(M3BP_NAME, "valid", "tune-chosen", cfg, summary)
    doc = save_frozen(M3BP_NAME, cfg, {"selection": "plan v5 E7: window (E2), same-day blend (E2 value rule), alpha per tertile if p < 0.05 on valid",
                                       "adaptive_test": test})
    meta = {"config": cfg, "adaptive_test": test, "valid_auc": float(summary["auc"]), "uniform_valid_auc": float(per_uniform["auc"].mean()),
            "run_info": run_info()}
    (TABLES / "behaviour_e7_m3bp_meta.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    print(table.round(5).to_string(index=False))
    print(f"chosen {cfg}; record '{frozen_marker(M3BP_NAME, doc['config_hash'])}' in docs/decisions.md before --final-dev")


def final_dev(reason):
    frozen = check_final_dev(M3BP_NAME, reason)
    cfg = frozen["config"]
    if cfg["m2_config_hash"] != load_frozen(M2_NAME)["config_hash"]:
        raise SystemExit("M2 changed after M3b+ was frozen")
    ctx = load_context("dev")
    scores, _, tert, used = build(ctx, cfg)
    out = ctx.truth[["imp_key", "news_id"]].assign(score=scores)
    per, summary = evaluate(ctx.truth, out)
    save_results(M3BP_NAME, "dev", out, per, summary)
    append_run(M3BP_NAME, "dev", "final-dev", cfg, summary, note=reason or "")
    if cfg["blend_a"] == 1.0:
        _, used, _ = same_day(ctx)    # coverage is reported on every part (plan v5 E2), even when the blend is not used
    meta = {"config": cfg, "metrics": summary.round(5).to_dict(), "same_day_coverage_dev": float((used > 0).mean()),
            "tertile_shares_dev": pd.Series(tert).value_counts(normalize=True).round(4).to_dict(), "run_info": run_info()}
    (TABLES / "behaviour_e7_m3bp_dev_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tune", action="store_true")
    mode.add_argument("--final-dev", action="store_true")
    ap.add_argument("--rerun-dev", metavar="REASON")
    args = ap.parse_args()
    tune() if args.tune else final_dev(args.rerun_dev)


if __name__ == "__main__":
    main()
