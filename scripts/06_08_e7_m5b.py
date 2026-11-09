"""E7b (plan v5): M5b, LightGBM LambdaRank + emotion in five pre-declared variants (V1 to V5), 3 seeds each, each against its reference
(the same features minus emotion, same hyperparameters and seeds). One variant is chosen by mean valid AUC; ties go to the simpler one.
The final M5b is the mean of the 3 seeds' predictions, run on dev once.

Reads:  the M4 feature tables (fit, valid), behaviours and impressions, behaviour_e2_choice.json, spec v5; dev features are built only in --final-dev.
Writes: --tune: data/processed/ltr_features_v5_{fit,valid}.parquet, boosters, results/frozen/M5b_lightgbm_emotion.json, valid files,
        results/tables/behaviour_e7_m5b_*.csv and _meta.json. --final-dev: dev files, a summary.csv row, a runs_log.csv row.
Run:    python scripts/06_08_e7_m5b.py --tune [--resume]
        python scripts/06_08_e7_m5b.py --final-dev    # once, after the frozen marker is in docs/decisions.md
"""
import argparse
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.behaviour import exposure_bucket, history_consistency, personal_match, window_profiles
from src.behaviour_runs import exposure_table, load_context
from src.config import (BEHAVIOUR_SPEC_VERSION, LTR_FEATURE_FILE, LTR_GRID, LTR_IMPORTANCE_IMPRESSIONS, M4_FEATURES, M4_NAME, M5B_NAME, PROCESSED,
                        RESULTS, SEED, TABLES)
from src.emotion_spec import load_spec
from src.evaluate import evaluate, load_truth, save_results
from src.lightgbm_ranker import build_features, check_saved_scores, file_sha, fit_or_load, importance_table, load_model, save_model_copy
from src.provenance import run_info
from src.run_registry import append_run, check_final_dev, frozen_marker, load_frozen, save_frozen
from src.click_signal import permute_within_groups
from src.subgroups import paired_family

NEW = ["consistency", "cong_short_personal", "exposure_bucket"]
BUCKET_CODE = {"new": 0, "few": 1, "many": 2}
V5_FILE = "ltr_features_v5_{part}.parquet"


def short_window(e2):
    """The E2 window, or last 10 (decay 0.9) when the current profile (50, 0.9) won (plan v5 E7)."""
    w = e2["window"]
    return (10, 0.9) if (w["size"], w["decay"]) == (50, 0.9) else (w["size"], w["decay"])


def add_columns(frame, part, short, expo, q, processed=PROCESSED):
    """The three new columns for one part, in harness order (time-causal: history column, strictly earlier exposure)."""
    ctx = load_context(part, processed, with_m2=False)
    if frame["imp_key"].tolist() != ctx.truth["imp_key"].tolist():
        raise ValueError(f"{part} feature rows are not in harness order")
    pair, _ = history_consistency(ctx.hist_rows, ctx.vectors)
    prof, _ = window_profiles(ctx.hist_rows, ctx.vectors, int(short[0]), float(short[1]))
    out = frame.copy()
    out["consistency"] = pair[ctx.groups].astype(np.float32)
    out["cong_short_personal"] = personal_match(ctx.groups, ctx.item_idx, prof, ctx.vectors).astype(np.float32)
    keys = pd.MultiIndex.from_frame(ctx.truth[["imp_key", "news_id"]])
    b = exposure_bucket(expo.reindex(keys).to_numpy(), q)
    out["exposure_bucket"] = np.array([BUCKET_CODE[x] for x in b], dtype=np.int8)
    return out


def variants(spec, m4_params):
    """{name: (features, params)} for V1 to V5 and their references, from spec v5."""
    v = spec["m5b_variants"]
    v1 = [*M4_FEATURES, *v["V1"]["add"]]
    v2 = [*v1, *v["V2"]["add"]]
    v3 = [*v2, *v["V3"]["add"]]
    reg = dict(v["V4"]["params"])
    mono = dict(reg, monotone_constraints=[1 if f in v["V5"]["monotone_increasing"] else 0 for f in v2])
    ref_b = [*M4_FEATURES, "exposure_bucket"]
    return {"V1": (v1, m4_params), "V2": (v2, m4_params), "V3": (v3, m4_params), "V4": (v2, reg), "V5": (v2, mono),
            "ref_M4": (list(M4_FEATURES), m4_params), "ref_M4_exposure": (ref_b, m4_params), "ref_M4_exposure_regularised": (ref_b, reg)}


REFS = {"V1": "ref_M4", "V2": "ref_M4_exposure", "V3": "ref_M4_exposure", "V4": "ref_M4_exposure_regularised", "V5": "ref_M4_exposure_regularised"}


def predict_mean(boosters, frame, features):
    return np.mean([b.predict(frame[features]) for b in boosters], axis=0)


def tune(resume):
    spec = load_spec(BEHAVIOUR_SPEC_VERSION)["behaviour"]
    e2 = json.loads((TABLES / "behaviour_e2_choice.json").read_text(encoding="utf-8"))
    expo, q = exposure_table(PROCESSED)
    frames = {}
    for part in ("fit", "valid"):
        frames[part] = add_columns(pd.read_parquet(PROCESSED / LTR_FEATURE_FILE.format(part=part)), part, short_window(e2), expo, q)
        frames[part].to_parquet(PROCESSED / V5_FILE.format(part=part), index=False)
    del expo
    fit, valid = frames["fit"], frames["valid"]
    sha = file_sha(*(PROCESSED / V5_FILE.format(part=p) for p in ("fit", "valid")))
    truth = load_truth("valid", PROCESSED)
    m4_frozen = load_frozen(M4_NAME)["config"]
    m4_params = {k: m4_frozen[k] for k in LTR_GRID}
    runs = variants(spec, m4_params)
    seeds = spec["m5b_variants"]["seeds"]
    rows, boosters = [], {}
    for name, (features, params) in runs.items():
        for seed in seeds:
            booster, info = fit_or_load(fit, valid, features, params, seed, "v5", sha, resume)
            boosters[(name, seed)] = (booster, info)
            per, summary = evaluate(truth, valid[["imp_key", "news_id"]].assign(score=booster.predict(valid[features])))
            rows.append({"variant": name, "seed": seed, "n_features": len(features), "best_iteration": info["best_iteration"],
                         **summary.round(6).to_dict()})
            print(f"{name} seed {seed}: AUC {summary['auc']:.5f}, {info['best_iteration']} rounds", flush=True)
    table = pd.DataFrame(rows)
    ref_auc = table.set_index(["variant", "seed"])["auc"]
    table["ref"] = table["variant"].map(REFS)
    table["gain_vs_ref"] = [r["auc"] - ref_auc[(r["ref"], r["seed"])] if isinstance(r["ref"], str) else np.nan for _, r in table.iterrows()]
    table.to_csv(TABLES / "behaviour_e7_m5b_variants.csv", index=False)
    mean = table[table["variant"].isin(REFS)].groupby("variant")["auc"].mean()
    tol, order = spec["m5b_variants"]["tie_tolerance"], spec["m5b_variants"]["simplicity_order"]
    best = mean.max()
    chosen = next(v for v in order if best - mean[v] < tol)
    features, params = runs[chosen]
    chosen_boosters = [boosters[(chosen, s)][0] for s in seeds]
    users = load_context("valid", with_m2=False).users_by_imp
    out = valid[["imp_key", "news_id"]].assign(score=predict_mean(chosen_boosters, valid, features))
    per, summary = evaluate(truth, out)
    ref_name = REFS[chosen]
    ref_out = valid[["imp_key", "news_id"]].assign(score=predict_mean([boosters[(ref_name, s)][0] for s in seeds], valid, runs[ref_name][0]))
    per_ref, _ = evaluate(truth, ref_out)
    per_m4 = pd.read_parquet(RESULTS / "per_impression" / f"{M4_NAME}__valid.parquet")
    fam = paired_family([(f"M5b ({chosen}, 3-seed mean) - its reference", per_ref, per), ("M5b - M4 (frozen)", per_m4, per)], users)
    # shuffled-label control: the chosen variant retrained on fit labels permuted within impressions
    fit_sh = fit.assign(label=permute_within_groups(pd.factorize(fit["imp_key"])[0], fit["label"].to_numpy(), SEED))
    b_sh, info_sh = fit_or_load(fit_sh, valid, features, params, SEED, "v5-shuffled-labels", sha, resume)
    _, sh_summary = evaluate(truth, valid[["imp_key", "news_id"]].assign(score=b_sh.predict(valid[features])))
    # permutation check: chosen seed-42 booster scored with its emotion columns permuted within impressions
    emo_cols = [f for f in features if f not in runs[ref_name][0]]
    from src.lightgbm_ranker import shuffle_within_impressions
    valid_p = shuffle_within_impressions(valid, emo_cols, SEED)
    _, perm_summary = evaluate(truth, valid[["imp_key", "news_id"]].assign(score=chosen_boosters[0].predict(valid_p[features])))
    fam = pd.concat([fam, pd.DataFrame([{"comparison": "control: retrained on shuffled labels", "metric": "auc", "mean_b": float(sh_summary["auc"])},
                                        {"comparison": "check: emotion columns permuted at scoring (seed 42)", "metric": "auc",
                                         "mean_b": float(perm_summary["auc"])}])], ignore_index=True)
    fam.to_csv(TABLES / "behaviour_e7_m5b_valid.csv", index=False)
    imp = importance_table(chosen_boosters[0], valid, features, LTR_IMPORTANCE_IMPRESSIONS, SEED)
    imp.to_csv(TABLES / "behaviour_e7_m5b_importance.csv", index=False)
    save_results(M5B_NAME, "valid", out, per, summary)
    shas = {str(s): save_model_copy(boosters[(chosen, s)][1], f"{M5B_NAME}_seed{s}") for s in seeds}
    cfg = {"variant": chosen, "features": features, "params": params, "seeds": seeds, "model_sha256": shas, "exposure_q_fit": q,
           "short_window": list(short_window(e2)), "m4_config_hash": load_frozen(M4_NAME)["config_hash"]}
    append_run(M5B_NAME, "valid", "tune-chosen", cfg, summary)
    doc = save_frozen(M5B_NAME, cfg, {"selection": "plan v5 E7: highest mean valid AUC over 3 seeds, ties (< 0.0005) to the simpler variant",
                                      "mean_valid_auc": mean.round(6).to_dict()})
    meta = {"chosen": chosen, "mean_valid_auc_by_variant": mean.round(6).to_dict(), "ensemble_valid_auc": float(summary["auc"]),
            "shuffled_label_valid_auc": float(sh_summary["auc"]), "permuted_emotion_valid_auc": float(perm_summary["auc"]), "run_info": run_info()}
    (TABLES / "behaviour_e7_m5b_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(table.round(5).to_string(index=False), fam.round(5).to_string(index=False), sep="\n")
    print(f"chosen {chosen}; record '{frozen_marker(M5B_NAME, doc['config_hash'])}' in docs/decisions.md before --final-dev")


def final_dev(reason):
    frozen = check_final_dev(M5B_NAME, reason)
    cfg = frozen["config"]
    if cfg["m4_config_hash"] != load_frozen(M4_NAME)["config_hash"]:
        raise SystemExit("M4 changed after M5b was frozen")
    boosters = [load_model(f"{M5B_NAME}_seed{s}", cfg["model_sha256"][str(s)]) for s in cfg["seeds"]]
    frame = build_features("dev", PROCESSED)
    check_saved_scores(frame, "dev")
    expo, _ = exposure_table(PROCESSED)
    frame = add_columns(frame, "dev", cfg["short_window"], expo, cfg["exposure_q_fit"])
    truth = load_truth("dev", PROCESSED)
    out = frame[["imp_key", "news_id"]].assign(score=predict_mean(boosters, frame, cfg["features"]))
    per, summary = evaluate(truth, out)
    save_results(M5B_NAME, "dev", out, per, summary)
    append_run(M5B_NAME, "dev", "final-dev", {k: cfg[k] for k in ("variant", "seeds", "params")}, summary, note=reason or "")
    print(summary.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tune", action="store_true")
    mode.add_argument("--final-dev", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rerun-dev", metavar="REASON")
    args = ap.parse_args()
    tune(args.resume) if args.tune else final_dev(args.rerun_dev)


if __name__ == "__main__":
    main()
