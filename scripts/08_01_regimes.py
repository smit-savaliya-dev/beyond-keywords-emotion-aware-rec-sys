"""Plan v7, Step 1: tree models in the two regimes. R1 STANDARD = previous-days-only behavioural features (primary); R2 REAL-TIME = M4v6 / M5v6.

  --build            previous-days-only (pd_) columns for every candidate row; fit / valid side files in the row order of ltr_v6_{part}.parquet
  --tune MODEL ...   Optuna, 8 trials on F3 + F4 (seed 42), the plan-v6 space and budget, resumable (sqlite study optuna_v7.db)
  --evaluate MODEL   best setting x seeds 42-44 x folds F1-F4 -> results/tables/v7_folds.csv; the F4 3-seed mean is the valid file
  --controls         shuffled labels on F4 with a fixed number of rounds (D92), R1 twin + R1 emotion + R2 twin + R2 emotion
  --select           Rule 8 in R1 (twin M4v6_R1) -> v7_selection.json; freezes M4v6_R1 and M5v6_R1
  --final-dev LABEL  retrain on days 9-14 with frozen settings, dev once, 3-seed mean (guarded by the decision log)
Models: T1_M4cols (old M4 columns), M4v6_R1 (R1 twin), M5v6_R1_F2 / _F3 / _F6 (R1 emotion). The R2 pair (M4v6, M5v6_F2) is read from plan v6.
Reads: ltr_v6_{fit,valid}.parquet, impressions, the plan-v6 study (R2 settings). Writes: data/processed/ltr_v7_*, results/tables/v7_*.
"""
import argparse
import importlib
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import M4_FEATURES, PROCESSED, SEED, TABLES
from src.evaluate import evaluate, load_truth, save_results
from src.lightgbm_ranker import build_features, file_sha, fit_or_load, train_ranker
from src.ltr_v6 import FOLDS, search_space, train_fixed, within_impression
from src.run_registry import append_run, check_final_dev, frozen_marker, load_frozen, save_frozen
from src.same_day import PD_COLS, PD_NORM, all_prev_day_features
from src.subgroups import fast_metrics

v6 = importlib.import_module("07_02_trees_v6")

PD_ALL = PROCESSED / "ltr_v7_pd_all.parquet"
STUDY = PROCESSED / "optuna_v7.db"
SEEDS = v6.SEEDS
N_TRIALS = v6.N_TRIALS
BASE_NORM = [f"{c}_{k}" for c in ("pop_ctr", "m1_score", "m2_score", "user_cat_affinity") for k in ("rank", "z")]
R1_TWIN = [*M4_FEATURES, *PD_COLS, *BASE_NORM, *PD_NORM]
MODELS = {"T1_M4cols": list(M4_FEATURES), "M4v6_R1": R1_TWIN, "M5v6_R1_F2": [*R1_TWIN, *v6.F2], "M5v6_R1_F3": [*R1_TWIN, *v6.F3],
          "M5v6_R1_F6": [*R1_TWIN, *v6.F2, "m3bp_score"]}
R2 = {"M4v6": v6.MODELS["M4v6"], "M5v6_F2": v6.MODELS["M5v6_F2"]}
SIMPLER = ["M5v6_R1_F2", "M5v6_R1_F6", "M5v6_R1_F3"]
FOLD_TABLE = TABLES / "v7_folds.csv"


def pd_path(part):
    return PROCESSED / f"ltr_v7_pd_{part}.parquet"


def attach_pd(keys, pd_all):
    """pd_ columns for rows `keys` (imp_key, news_id) in their order, plus the within-list rank / z of pd_ctr_24h."""
    p = pd_all.reindex(pd.MultiIndex.from_arrays([keys["imp_key"].astype(str).to_numpy(), keys["news_id"].astype(str).to_numpy()]))
    if p[PD_COLS[0]].isna().any():
        raise ValueError("rows without pd_ features")
    out = {c: p[c].to_numpy(np.float32) for c in PD_COLS}
    norm = within_impression(pd.DataFrame({"imp_key": keys["imp_key"].to_numpy(), "pd_ctr_24h": out["pd_ctr_24h"]}), ["pd_ctr_24h"])
    out.update({c: norm[c].to_numpy(np.float32) for c in PD_NORM})
    return out


def load_pd_all():
    return pd.read_parquet(PD_ALL).set_index(["imp_key", "news_id"])


def build():
    if not PD_ALL.exists():
        all_prev_day_features(PROCESSED).to_parquet(PD_ALL, index=False)
        print("built", PD_ALL, flush=True)
    pd_all = load_pd_all()
    for p in ("fit", "valid"):
        keys = pd.read_parquet(v6.part_path(p), columns=["imp_key", "news_id", "day"])
        cols = {"imp_key": keys["imp_key"].astype(str).to_numpy(), "news_id": keys["news_id"].astype(str).to_numpy(), "day": keys["day"].to_numpy(),
                **attach_pd(keys, pd_all)}
        v6.write_cols(cols, pd_path(p))
        print("built", pd_path(p), flush=True)


def load_rows(features, test=None, upto=None):
    """Rows of fit + valid with `features` (pd_ columns joined from the side files). test: a fold's test day -> (train, test);
    upto: all rows up to that day -> one frame."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    last = pd.Timestamp(test if test is not None else upto)
    pdc = [c for c in features if c in PD_COLS or c in PD_NORM]
    base = list(dict.fromkeys(["imp_key", "news_id", "label", "day", *[c for c in features if c not in pdc]]))
    tables = []
    for p in ("fit", "valid"):
        t = pq.read_table(v6.part_path(p), columns=base, filters=[("day", "<=", last)])
        if pdc:
            s = pq.read_table(pd_path(p), columns=["imp_key", *pdc], filters=[("day", "<=", last)])
            if t.num_rows != s.num_rows or not (t["imp_key"].to_numpy(zero_copy_only=False) == s["imp_key"].to_numpy(zero_copy_only=False)).all():
                raise ValueError(f"{p}: pd side file is not in row order")
            for c in pdc:
                t = t.append_column(c, s[c])
        tables.append(t)
    t = pa.concat_tables(tables, promote_options="permissive")
    del tables
    if test is None:
        f = t.to_pandas(split_blocks=True)
        f["day"] = pd.to_datetime(f["day"]).dt.normalize()
        return f
    is_test = pc.equal(t["day"], pa.scalar(last, type=t["day"].type))
    out = []
    for mask in (pc.invert(is_test), is_test):
        f = t.filter(mask).to_pandas(split_blocks=True)
        f["day"] = pd.to_datetime(f["day"]).dt.normalize()
        out.append(f)
    return out[0], out[1]


def data_sha():
    return file_sha(v6.part_path("fit"), v6.part_path("valid"), pd_path("fit"), pd_path("valid"))


def tune(models):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for name in models:
        folds = {k: load_rows(MODELS[name], test=FOLDS[k]) for k in ("F3", "F4")}
        study = optuna.create_study(study_name=name, storage=f"sqlite:///{STUDY.as_posix()}", direction="maximize", load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(seed=SEED))

        def objective(trial):
            params = search_space(trial, False)
            return float(np.mean([train_ranker(tr, te, MODELS[name], params, SEED)[3] for tr, te in folds.values()]))
        left = N_TRIALS - len([t for t in study.trials if t.state.name == "COMPLETE"])
        if left > 0:
            study.optimize(objective, n_trials=left)
        del folds
        print(name, "best", round(study.best_value, 5), study.best_params, flush=True)


def best_params(name):
    import optuna
    if name in R2:
        return v6.best_params(name)
    return optuna.load_study(study_name=name, storage=f"sqlite:///{STUDY.as_posix()}").best_params


def evaluate_model(name):
    sha, params, feats = data_sha(), best_params(name), MODELS[name]
    rows = []
    for fold, day in FOLDS.items():
        tr, te = load_rows(feats, test=day)
        preds = []
        for seed in SEEDS:
            booster, info = fit_or_load(tr, te, feats, params, seed, f"v7-{fold}", sha, resume=True)
            p = booster.predict(te[feats])
            preds.append(p)
            auc = float(np.nanmean(fast_metrics(te["imp_key"].to_numpy(), te["label"].to_numpy(), p)["auc"]))
            rows.append({"model": name, "fold": fold, "seed": seed, "auc": auc, "best_iteration": info["best_iteration"]})
            print(rows[-1], flush=True)
        fast_metrics(te["imp_key"].to_numpy(), te["label"].to_numpy(), np.mean(preds, axis=0)).to_parquet(
            PROCESSED / f"ltr_v7_eval_{name}_{fold}.parquet", index=False)
        if fold == "F4":
            out = te[["imp_key", "news_id"]].assign(score=np.mean(preds, axis=0))
            per_h, summ = evaluate(load_truth("valid", PROCESSED), out)
            save_results(name, "valid", out, per_h, summ)
            append_run(name, "valid", "rolling-F4-3seed-mean", {"params": params, "seeds": SEEDS, "plan": "v7"}, summ)
        del tr, te
    t = pd.DataFrame(rows)
    old = pd.read_csv(FOLD_TABLE) if FOLD_TABLE.exists() else pd.DataFrame(columns=t.columns)
    pd.concat([old[old["model"] != name], t], ignore_index=True).to_csv(FOLD_TABLE, index=False)


def mean_rounds(name):
    t = pd.read_csv(FOLD_TABLE if name in MODELS else TABLES / "v6_trees_folds.csv")
    return int(round(t[t["model"] == name]["best_iteration"].mean()))


def controls():
    """D92: labels permuted within impressions on F4 train, fixed rounds (the model's mean best iteration), scored on F4 with true labels."""
    from src.click_signal import permute_within_groups
    sel = json.loads((TABLES / "v7_selection.json").read_text(encoding="utf-8"))["chosen"]
    rows = []
    for name, feats in (("M4v6_R1", MODELS["M4v6_R1"]), (sel, MODELS[sel]), ("M4v6", R2["M4v6"]), ("M5v6_F2", R2["M5v6_F2"])):
        tr, te = load_rows(feats, test=FOLDS["F4"])
        tr = tr.assign(label=permute_within_groups(pd.factorize(tr["imp_key"])[0], tr["label"].to_numpy(), SEED))
        rounds = mean_rounds(name)
        booster = train_fixed(tr, feats, best_params(name), SEED, rounds)
        auc = float(np.nanmean(fast_metrics(te["imp_key"].to_numpy(), te["label"].to_numpy(), booster.predict(te[feats]))["auc"]))
        rows.append({"model": name, "regime": "R2" if name in R2 else "R1", "rounds": rounds, "shuffled_label_valid_auc": auc})
        print(rows[-1], flush=True)
        del tr, te
    pd.DataFrame(rows).to_csv(TABLES / "v7_shuffled_controls.csv", index=False)


def select():
    t = pd.read_csv(FOLD_TABLE)
    m = t.groupby(["model", "fold"])["auc"].mean().unstack()
    noise = float(t[t["model"] == "M4v6_R1"].groupby("fold")["auc"].std().mean())
    rows = []
    for v in SIMPLER:
        wins = int((m.loc[v, ["F2", "F3", "F4"]] > m.loc["M4v6_R1", ["F2", "F3", "F4"]]).sum())
        gain = float((m.loc[v] - m.loc["M4v6_R1"]).mean())
        rows.append({"variant": v, "mean_auc": float(m.loc[v].mean()), "gain_vs_twin": gain, "wins_F2_F4": wins, "noise_sd": noise,
                     "eligible": wins >= 2 and gain > noise, **{f"gain_{f}": float(m.loc[v, f] - m.loc["M4v6_R1", f]) for f in FOLDS}})
    r = pd.DataFrame(rows)
    elig = r[r["eligible"]]
    if len(elig):
        best = elig["mean_auc"].max()
        chosen = next(v for v in SIMPLER if v in set(elig["variant"]) and best - float(elig.set_index("variant").loc[v, "mean_auc"]) < 0.0005)
    else:
        chosen = "M5v6_R1_F2"
    r.to_csv(TABLES / "v7_rule8.csv", index=False)
    sel = {"chosen": chosen, "passed_rule": bool(len(elig)), "twin_noise_sd": noise}
    (TABLES / "v7_selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    print(r.round(5).to_string(index=False), sel, sep="\n")
    for name, label in (("M4v6_R1", "M4v6_R1"), (chosen, "M5v6_R1")):
        rounds = t[t["model"] == name].groupby("seed")["best_iteration"].mean().round().astype(int).to_dict()
        doc = save_frozen(label, {"source": name, "features": MODELS[name], "params": best_params(name), "seeds": SEEDS,
                                  "rounds_by_seed": {str(k): int(v) for k, v in rounds.items()}, "train_days": "9-14 Nov (fit + valid)",
                                  "regime": "R1 (previous days only)"},
                          {"selection": "plan v7 rule 8" if label == "M5v6_R1" else "plan v7 R1 twin", "rule8_passed": sel["passed_rule"]})
        print(f"record '{frozen_marker(label, doc['config_hash'])}' in docs/decisions.md", flush=True)


def final_dev(label, reason):
    frozen = check_final_dev(label, reason)
    cfg = frozen["config"]
    feats = cfg["features"]
    frame = load_rows(feats, upto="2019-11-14")
    base = build_features("dev", PROCESSED)
    extra = pd.read_parquet(v6.EXTRA).set_index(["imp_key", "news_id"]).reindex(pd.MultiIndex.from_frame(base[["imp_key", "news_id"]].astype(str)))
    cols = v6.part_frame("dev", base, extra, v6.emotion_inputs())
    del extra
    for c in v6.EXTRA_COLS + ["ctr_24h_rank", "ctr_24h_z"]:      # R1: drop every same-day column before anything else sees the frame
        cols.pop(c, None)
    cols.update(attach_pd(pd.DataFrame({"imp_key": cols["imp_key"], "news_id": cols["news_id"]}), load_pd_all()))
    dev = pd.DataFrame({c: cols[c] for c in ["imp_key", "news_id", *feats]})
    del cols
    preds = [train_fixed(frame, feats, cfg["params"], int(s), cfg["rounds_by_seed"][str(s)]).predict(dev[feats]) for s in cfg["seeds"]]
    out = dev[["imp_key", "news_id"]].assign(score=np.mean(preds, axis=0))
    per, summ = evaluate(load_truth("dev", PROCESSED), out)
    save_results(label, "dev", out, per, summ)
    append_run(label, "dev", "final-dev", {"source": cfg["source"], "seeds": cfg["seeds"], "plan": "v7"}, summ, note=reason or "")
    print(label, summ.round(4).to_dict(), flush=True)


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
    if a.select:
        select()
    if a.controls:
        controls()
    for m in a.final_dev:
        final_dev(m, a.rerun_dev)


if __name__ == "__main__":
    main()
