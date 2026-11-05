"""Stages 3-4 (plan v6): NRMS M6 and the emotion variants M7a/b/c and M7ctrl. Trained on fit (days 9-13) with early stopping on valid
(fold F4); 3 seeds each; checkpointed and resumable. --final-dev scores dev once per model with the 3 seeds' best checkpoints (mean).

Reads:  news titles, emotion tables, fit and valid (and for --final-dev dev) candidates and histories.
Writes: data/processed/nrms_runs/<model>_s<seed>/; results/tables/v6_nrms_*.csv; valid (and dev) scores and per-impression files.
Run:    python scripts/07_03_nrms.py --train M6 M7a M7b M7c      # any subset; resumes finished epochs
        python scripts/07_03_nrms.py --ctrl M7b                  # the control for the selected variant
        python scripts/07_03_nrms.py --final-dev M6 M7b M7ctrl   # once, after the frozen markers are in docs/decisions.md
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401
from src.config import RESULTS, SEED, TABLES
from src.evaluate import evaluate, save_results
from src.nrms_runs import RUN_DIR, Tables, news_tables, part_data, score_part, train_run
from src.run_registry import append_run, check_final_dev, frozen_marker, save_frozen

SEEDS = [42, 43, 44]
MODES = {"M6": "none", "M7a": "a", "M7b": "b", "M7c": "c"}
NAMES = {"M6": "M6_nrms", "M7a": "M7a_nrms_emotion_item", "M7b": "M7b_nrms_emotion_gated", "M7c": "M7c_nrms_emotion_both", "M7ctrl": "M7ctrl_nrms_permuted"}


def setup():
    tokens, emb, emo, pct = news_tables()
    fit, valid = part_data("fit"), part_data("valid")
    fit_items = np.unique(np.concatenate(fit.cands))
    perm = np.random.default_rng(SEED).permutation(len(tokens))     # fixed, fit-defined item swap for M7ctrl
    return tokens, emb, emo, pct, fit, valid, fit_items, perm


def train(models, ctrl_of=None):
    dev = torch.device("cuda")
    tokens, emb, emo, pct, fit, valid, fit_items, perm = setup()
    rows = []
    jobs = [(m, MODES[m], None) for m in models] + ([("M7ctrl", MODES[ctrl_of], perm)] if ctrl_of else [])
    for name, mode, p in jobs:
        tab = Tables(tokens, emo, pct, fit_items, dev, perm=p)
        if p is not None:
            fit, valid = part_data("fit", perm=p), part_data("valid", perm=p)
        seed_scores = []
        for seed in SEEDS:
            model, hist = train_run(f"{name}_s{seed}", mode, seed, fit, np.arange(len(fit.cands)), valid, np.arange(len(valid.cands)), tab, emb, dev)
            s = score_part(model, valid, tab, dev)
            seed_scores.append(s)
            per, summ = evaluate(valid.ctx.truth, valid.ctx.truth[["imp_key", "news_id"]].assign(score=s))
            best = max(hist, key=lambda h: h["valid_auc"])
            rows.append({"model": name, "seed": seed, "best_epoch": best["epoch"], "epochs_run": len(hist), **summ.round(6).to_dict()})
            pd.DataFrame(hist).assign(model=name, seed=seed).to_csv(RUN_DIR / f"{name}_s{seed}" / "curve.csv", index=False)
            print(rows[-1], flush=True)
            del model
            torch.cuda.empty_cache()
        mean = np.mean(seed_scores, axis=0)
        out = valid.ctx.truth[["imp_key", "news_id"]].assign(score=mean)
        per, summ = evaluate(valid.ctx.truth, out)
        save_results(NAMES[name], "valid", out, per, summ)
        append_run(NAMES[name], "valid", "train-3seed-mean", {"mode": mode, "seeds": SEEDS, "control": p is not None}, summ)
    t = pd.DataFrame(rows)
    path = TABLES / "v6_nrms_seeds_valid.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old[~old["model"].isin(t["model"])] if len(old) else old, t], ignore_index=True).to_csv(path, index=False)


def freeze(name, mode, control):
    doc = save_frozen(NAMES[name], {"mode": mode, "seeds": SEEDS, "control": control, "trained_on": "fit (days 9-13), early stopping on valid"},
                      {"selection": "plan v6 stage 3/4"})
    print(f"record '{frozen_marker(NAMES[name], doc['config_hash'])}' in docs/decisions.md before --final-dev")


def final_dev(models, ctrl_mode, reason):
    dev = torch.device("cuda")
    tokens, emb, emo, pct = news_tables()
    fit_items = np.unique(np.concatenate(part_data("fit").cands))
    perm = np.random.default_rng(SEED).permutation(len(tokens))
    from src.nrms import NRMS
    for name in models:
        data = part_data("dev", perm=perm if name == "M7ctrl" else None)
        frozen = check_final_dev(NAMES[name], reason)
        mode = frozen["config"]["mode"]
        tab = Tables(tokens, emo, pct, fit_items, dev, perm=perm if name == "M7ctrl" else None)
        scores = []
        for seed in SEEDS:
            run = RUN_DIR / f"{name}_s{seed}"
            hist = json.loads((run / "log.json").read_text(encoding="utf-8"))
            best = max(hist, key=lambda h: h["valid_auc"])["epoch"]
            model = NRMS(emb, mode).to(dev)
            model.load_state_dict(torch.load(run / f"epoch{best}.pt", map_location=dev, weights_only=False)["model"])
            scores.append(score_part(model, data, tab, dev))
        out = data.ctx.truth[["imp_key", "news_id"]].assign(score=np.mean(scores, axis=0))
        per, summ = evaluate(data.ctx.truth, out)
        save_results(NAMES[name], "dev", out, per, summ)
        append_run(NAMES[name], "dev", "final-dev", frozen["config"], summ, note=reason or "")
        print(name, summ.round(4).to_dict(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--ctrl", metavar="VARIANT")
    ap.add_argument("--freeze", nargs="*", default=[])
    ap.add_argument("--final-dev", nargs="*", default=[])
    ap.add_argument("--rerun-dev", metavar="REASON")
    args = ap.parse_args()
    if args.train or args.ctrl:
        train(args.train, args.ctrl)
    for name in args.freeze:
        freeze(name, MODES.get(name, "none") if name != "M7ctrl" else json.loads((TABLES / "v6_nrms_selection.json").read_text())["selected_mode"],
               name == "M7ctrl")
    if args.final_dev:
        final_dev(args.final_dev, None, args.rerun_dev)


if __name__ == "__main__":
    main()
