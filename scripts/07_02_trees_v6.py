"""Stage 2 (plan v6): M4v6 (no emotion) and the M5v6 variants F1-F6 (user-relative emotion), LightGBM LambdaRank on rolling-origin folds.

  --build            feature frame for fit + valid (data/processed/ltr_v6_fitvalid.parquet; extra features cached for all parts)
  --tune MODEL ...   Optuna, 8 trials on F3 + F4 (seed 42), resumable (sqlite study)
  --evaluate MODEL   best setting x seeds 42-44 x folds F1-F4; fold results and the F4 (valid) 3-seed-mean prediction
  --controls         every model retrained on F4 with labels permuted within impressions
  --select           Rule 8; writes v6_trees_selection.json
  --final-dev MODEL  retrain on days 9-14 (rounds = mean best iteration over folds), dev once, 3-seed mean
Reads: M4 feature tables, impressions, behaviours, news, emotion tables, saved M2 scores. Writes: results/tables/v6_trees_*.csv, results files.
"""
import argparse
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.behaviour import residual_vectors
from src.behaviour_runs import emotion_score, fused_scores, load_context
from src.config import FROZEN_DIR, LTR_FEATURE_FILE, M4_FEATURES, M4_NAME, PROCESSED, RESULTS, SEED, TABLES
from src.emotion import GOEMOTIONS_LABELS
from src.evaluate import evaluate, load_truth, save_results
from src.lightgbm_ranker import build_features, file_sha, fit_or_load, train_ranker
from src.ltr_v6 import FOLDS, NORM_COLS, emotion_columns, extra_features, fold_masks, persona_model, search_space, train_fixed, within_impression
from src.run_registry import append_run, check_final_dev, frozen_marker, load_frozen, save_frozen
from src.subgroups import fast_metrics

FRAME = PROCESSED / "ltr_v6_fitvalid.parquet"
EXTRA = PROCESSED / "ltr_v6_extra.parquet"
STUDY = PROCESSED / "optuna_v6.db"
SEEDS = [42, 43, 44]
N_TRIALS = 8
EXTRA_COLS = ["ctr_1h", "shows_1h", "ctr_6h", "shows_6h", "ctr_24h", "shows_24h", "exposure_log", "age_hours", "user_log_clicks", "user_log_cat_share"]
NORM = [f"{c}_{k}" for c in NORM_COLS for k in ("rank", "z")]
M4V6 = [*M4_FEATURES, *EXTRA_COLS, *NORM]
F1 = [f"{b}{s}" for b in ("cong_pct_cos", "cong_six_negkl", "cong_valence_absdist") for s in ("", "_personal", "_userspecific")]
F2 = [*F1, "consistency", "consistency_x_personal", "persona"]
F3 = [*F2, "resid_cong"]
F4 = [*F3, *[f"prod_{j:02d}" for j in range(28)]]
MODELS = {"M4v6": M4V6, "M5v6_F1": [*M4V6, *F1], "M5v6_F2": [*M4V6, *F2], "M5v6_F3": [*M4V6, *F3], "M5v6_F4": [*M4V6, *F4],
          "M5v6_F5": [*M4V6, "m3bp_score"], "M5v6_F6": [*M4V6, *F2, "m3bp_score"]}
SIMPLER = ["M5v6_F5", "M5v6_F1", "M5v6_F2", "M5v6_F6", "M5v6_F3", "M5v6_F4"]
DROP = ["go_pct_excitement", "go_pct_joy", "go_pct_optimism", "go_pct_nervousness", "go_pct_annoyance", "go_pct_fear", "go_vad_valence",
        "go_vad_dominance", "vad_valence", "nrc_trust", "go_dist_optimism", "go_dist_nervousness", "go_dist_fear", "go_intensity", "go_emotion_mass"]


def emotion_inputs(processed=PROCESSED):
    """Persona model, residual vectors, raw28 and valence per news row (all fit-defined, label-free)."""
    fit = load_context("fit", processed, with_m2=False)
    km, mu, sd = persona_model(fit, k=json.loads((TABLES / "behaviour_e8_meta.json").read_text(encoding="utf-8"))["k"])
    news = pd.read_parquet(processed / "news.parquet", columns=["news_id", "category"])
    cats = news.set_index("news_id")["category"].astype(str).reindex(fit.news_ids).to_numpy()
    resid = residual_vectors(fit.vectors, cats, fit.item_idx)
    raw = pd.read_parquet(processed / "news_emotion_raw28_v2.parquet", columns=["news_id", *GOEMOTIONS_LABELS])
    val = pd.read_parquet(processed / "news_emotion_features_v2.parquet", columns=["news_id", "go_vad_valence"])
    return km, mu, sd, resid, raw[GOEMOTIONS_LABELS].to_numpy(np.float64), val["go_vad_valence"].to_numpy(np.float64)


def part_frame(part, base, extra, inputs, processed=PROCESSED):
    """One part's v6 frame in harness order: M4 features + extra + within-impression norms + emotion columns + M3b+ score."""
    km, mu, sd, resid, raw28, valence = inputs
    ctx = load_context(part, processed, with_m2=True)
    if base["imp_key"].tolist() != ctx.truth["imp_key"].tolist():
        raise ValueError(f"{part} rows are not in harness order")
    f = base.drop(columns=[c for c in DROP if c in base.columns]).reset_index(drop=True)
    keys = pd.MultiIndex.from_frame(f[["imp_key", "news_id"]].astype(str))
    e = extra.reindex(keys)
    for c in EXTRA_COLS:
        f[c] = e[c].to_numpy()
    f = pd.concat([f, within_impression(f), emotion_columns(ctx, km, mu, sd, resid, raw28, valence)], axis=1)
    f["m3bp_score"] = fused_scores(ctx, emotion_score(ctx, 50, 1.0), 0.8).astype(np.float32)
    return f


def build():
    if not EXTRA.exists():
        extra_features(PROCESSED).to_parquet(EXTRA, index=False)
    extra = pd.read_parquet(EXTRA).set_index(["imp_key", "news_id"])
    inputs = emotion_inputs()
    frames = [part_frame(p, pd.read_parquet(PROCESSED / LTR_FEATURE_FILE.format(part=p)), extra, inputs) for p in ("fit", "valid")]
    pd.concat(frames, ignore_index=True).to_parquet(FRAME, index=False)
    print("built", FRAME, flush=True)


def load_frame():
    f = pd.read_parquet(FRAME)
    f["day"] = pd.to_datetime(f["day"]).dt.normalize()
    return f


def fold_data(frame, fold):
    tr, te = fold_masks(frame["day"].to_numpy(), fold)
    return frame[tr].reset_index(drop=True), frame[te].reset_index(drop=True)


def tune(models):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    frame = load_frame()
    folds = {k: fold_data(frame, k) for k in ("F3", "F4")}
    for name in models:
        study = optuna.create_study(study_name=name, storage=f"sqlite:///{STUDY.as_posix()}", direction="maximize", load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        strong = name == "M5v6_F4"

        def objective(trial):
            params = search_space(trial, strong)
            aucs = [train_ranker(tr, te, MODELS[name], params, SEED)[3] for tr, te in folds.values()]
            return float(np.mean(aucs))
        left = N_TRIALS - len([t for t in study.trials if t.state.name == "COMPLETE"])
        if left > 0:
            study.optimize(objective, n_trials=left)
        print(name, "best", round(study.best_value, 5), study.best_params, flush=True)


def best_params(name):
    import optuna
    return optuna.load_study(study_name=name, storage=f"sqlite:///{STUDY.as_posix()}").best_params


def evaluate_model(name):
    frame = load_frame()
    sha = file_sha(FRAME)
    params = best_params(name)
    rows = []
    for fold in FOLDS:
        tr, te = fold_data(frame, fold)
        preds = []
        for seed in SEEDS:
            booster, info = fit_or_load(tr, te, MODELS[name], params, seed, f"v6-{fold}", sha, resume=True)
            p = booster.predict(te[MODELS[name]])
            preds.append(p)
            auc = float(np.nanmean(fast_metrics(te["imp_key"].to_numpy(), te["label"].to_numpy(), p)["auc"]))
            rows.append({"model": name, "fold": fold, "seed": seed, "auc": auc, "best_iteration": info["best_iteration"]})
            print(rows[-1], flush=True)
        per = fast_metrics(te["imp_key"].to_numpy(), te["label"].to_numpy(), np.mean(preds, axis=0))
        per.to_parquet(PROCESSED / f"ltr_v6_eval_{name}_{fold}.parquet", index=False)
        if fold == "F4":
            truth = load_truth("valid", PROCESSED)
            out = te[["imp_key", "news_id"]].assign(score=np.mean(preds, axis=0))
            per_h, summ = evaluate(truth, out)
            save_results(name, "valid", out, per_h, summ)
            append_run(name, "valid", "rolling-F4-3seed-mean", {"params": params, "seeds": SEEDS}, summ)
    t = pd.DataFrame(rows)
    path = TABLES / "v6_trees_folds.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=t.columns)
    pd.concat([old[old["model"] != name], t], ignore_index=True).to_csv(path, index=False)


def controls():
    from src.click_signal import permute_within_groups
    frame = load_frame()
    sha = file_sha(FRAME)
    tr, te = fold_data(frame, "F4")
    tr = tr.assign(label=permute_within_groups(pd.factorize(tr["imp_key"])[0], tr["label"].to_numpy(), SEED))
    rows = []
    for name in MODELS:
        booster, info = fit_or_load(tr, te, MODELS[name], best_params(name), SEED, "v6-F4-shuffled-labels", sha, resume=True)
        auc = float(np.nanmean(fast_metrics(te["imp_key"].to_numpy(), te["label"].to_numpy(), booster.predict(te[MODELS[name]]))["auc"]))
        rows.append({"model": name, "shuffled_label_valid_auc": auc, "rounds": info["best_iteration"]})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(TABLES / "v6_trees_shuffled_controls.csv", index=False)


def select():
    t = pd.read_csv(TABLES / "v6_trees_folds.csv")
    m = t.groupby(["model", "fold"])["auc"].mean().unstack()
    noise = float(t[t["model"] == "M4v6"].groupby("fold")["auc"].std().mean())
    rows = []
    for v in SIMPLER:
        if v not in m.index:
            continue
        wins = int((m.loc[v, ["F2", "F3", "F4"]] > m.loc["M4v6", ["F2", "F3", "F4"]]).sum())
        gain = float((m.loc[v] - m.loc["M4v6"]).mean())
        rows.append({"variant": v, "mean_auc": float(m.loc[v].mean()), "gain_vs_twin": gain, "wins_F2_F4": wins, "noise_sd": noise,
                     "eligible": wins >= 2 and gain > noise, **{f"gain_{f}": float(m.loc[v, f] - m.loc["M4v6", f]) for f in FOLDS}})
    r = pd.DataFrame(rows)
    elig = r[r["eligible"]]
    if len(elig):
        best = elig["mean_auc"].max()
        chosen = next(v for v in SIMPLER if v in set(elig["variant"]) and best - float(elig.set_index("variant").loc[v, "mean_auc"]) < 0.0005)
        passed = True
    else:
        chosen, passed = "M5v6_F5", False
    r.to_csv(TABLES / "v6_trees_rule8.csv", index=False)
    sel = {"chosen": chosen, "passed_rule": passed, "twin_noise_sd": noise, "m4v6_mean_auc": float(m.loc["M4v6"].mean())}
    (TABLES / "v6_trees_selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    print(r.round(5).to_string(index=False), sel, sep="\n")
    for name, label in (("M4v6", "M4v6"), (chosen, "M5v6")):
        rounds = t[t["model"] == name].groupby("seed")["best_iteration"].mean().round().astype(int).to_dict()
        doc = save_frozen(label, {"source": name, "features": MODELS[name], "params": best_params(name), "seeds": SEEDS,
                                  "rounds_by_seed": {str(k): int(v) for k, v in rounds.items()}, "train_days": "9-14 Nov (fit + valid)"},
                          {"selection": "plan v6 rule 8" if label == "M5v6" else "plan v6 twin", "rule8_passed": passed})
        print(f"record '{frozen_marker(label, doc['config_hash'])}' in docs/decisions.md")


def final_dev(label, reason):
    frozen = check_final_dev(label, reason)
    cfg = frozen["config"]
    frame = load_frame()
    extra = pd.read_parquet(EXTRA).set_index(["imp_key", "news_id"])
    dev = part_frame("dev", build_features("dev", PROCESSED), extra, emotion_inputs())
    preds = [train_fixed(frame, cfg["features"], cfg["params"], int(s), cfg["rounds_by_seed"][str(s)]).predict(dev[cfg["features"]]) for s in cfg["seeds"]]
    truth = load_truth("dev", PROCESSED)
    out = dev[["imp_key", "news_id"]].assign(score=np.mean(preds, axis=0))
    per, summ = evaluate(truth, out)
    save_results(label, "dev", out, per, summ)
    append_run(label, "dev", "final-dev", {"source": cfg["source"], "seeds": cfg["seeds"]}, summ, note=reason or "")
    print(label, summ.round(4).to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--tune", nargs="*", default=[])
    ap.add_argument("--evaluate", nargs="*", default=[])
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--final-dev", nargs="*", default=[])
    ap.add_argument("--rerun-dev", metavar="REASON")
    a = ap.parse_args()
    if a.build:
        build()
    if a.tune:
        tune(a.tune)
    for m in a.evaluate:
        evaluate_model(m)
    if a.controls:
        controls()
    if a.select:
        select()
    for m in a.final_dev:
        final_dev(m, a.rerun_dev)


if __name__ == "__main__":
    main()
