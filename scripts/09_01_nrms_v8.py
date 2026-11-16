"""Plan v8 (docs/experiments_plan_v8.md, D105): improving the neural ranker + emotion. Valid only; dev is never read here.

M7d: late fusion of each M6 seed with the personal match (M3 recipe), alpha cross-fitted over two user halves of valid.
M7e: NRMS + trained personal-match head (fit-standardised input, emotion learning rate 1e-2). M7f: M7e + item-emotion projection.

Reads:  M6 checkpoints (data/processed/nrms_runs/M6_s*), news and emotion tables, fit and valid candidates and histories,
        results/tables/v6_nrms_seeds_valid.csv (M6 seed rows and noise).
Writes: data/processed/nrms_runs/<model>_s<seed>/ (+ valid_scores.npy per seed); results/tables/v8_nrms_seeds_valid.csv,
        v8_m7d_alpha_grid.csv, v8_nrms_rule8.csv, v8_nrms_selection.json; valid scores/per-impression files of each variant.
Run:    python scripts/09_01_nrms_v8.py --fusion                # M7d and its permuted control (minutes)
        python scripts/09_01_nrms_v8.py --train M7e M7f         # about one hour per variant (3 seeds), resumable
        python scripts/09_01_nrms_v8.py --ctrl M7e              # permuted-emotion control of a trained variant
        python scripts/09_01_nrms_v8.py --select
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401
from src.behaviour import personal_match, window_profiles
from src.config import SEED, TABLES
from src.evaluate import evaluate, save_results
from src.fusion import fuse, minmax_within
from src.nrms import NRMS
from src.nrms_runs import RUN_DIR, Tables, news_tables, part_data, personal_match_stats, score_part, train_run
from src.run_registry import append_run
from src.subgroups import fast_metrics

SEEDS = [42, 43, 44]
MODES = {"M7e": "e", "M7f": "f"}
NAMES = {"M7d": "M7d_nrms_emotion_fusion", "M7e": "M7e_nrms_emotion_personal_head", "M7f": "M7f_nrms_emotion_head_item",
         "M7d_ctrl": "M7d_ctrl_permuted", "M7e_ctrl": "M7e_ctrl_permuted", "M7f_ctrl": "M7f_ctrl_permuted"}
ALPHAS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]
EMO_LR = 1e-2
SEEDS_TABLE = TABLES / "v8_nrms_seeds_valid.csv"
ORDER = ["M7d", "M7e", "M7f"]                     # simplest first (tie rule)


def setup(perm=None):
    tokens, emb, emo, pct = news_tables()
    fit, valid = part_data("fit", perm=perm), part_data("valid", perm=perm)
    fit_items = np.unique(np.concatenate(fit.cands))
    return tokens, emb, emo, pct, fit, valid, fit_items


def control_perm(n_items):
    return np.random.default_rng(SEED).permutation(n_items)       # the same item swap as M7ctrl (plan v6)


def add_seed_rows(rows):
    t = pd.DataFrame(rows)
    old = pd.read_csv(SEEDS_TABLE) if SEEDS_TABLE.exists() else pd.DataFrame()
    if len(old):
        old = old[~old.set_index(["model", "seed"]).index.isin(t.set_index(["model", "seed"]).index)]
    pd.concat([old, t], ignore_index=True).to_csv(SEEDS_TABLE, index=False)


def summary_row(name, seed, truth, scores, extra=None):
    _, summ = evaluate(truth, truth[["imp_key", "news_id"]].assign(score=scores))
    return {"model": name, "seed": seed, **(extra or {}), **summ.round(6).to_dict()}


def save_mean(name, truth, seed_scores, config):
    out = truth[["imp_key", "news_id"]].assign(score=np.mean(seed_scores, axis=0))
    per, summ = evaluate(truth, out)
    save_results(NAMES[name], "valid", out, per, summ)
    append_run(NAMES[name], "valid", "v8-3seed-mean", config, summ)
    print(NAMES[name], summ.round(4).to_dict(), flush=True)


# ------------------------------------------------------------------ M6 per-seed valid scores (no training)
def m6_seed_scores(valid, tables, emb, dev):
    out = {}
    for seed in SEEDS:
        run = RUN_DIR / f"M6_s{seed}"
        cache = run / "valid_scores.npy"
        if cache.exists():
            out[seed] = np.load(cache)
            continue
        hist = json.loads((run / "log.json").read_text(encoding="utf-8"))
        best = max(hist, key=lambda h: h["valid_auc"])["epoch"]
        model = NRMS(emb, "none").to(dev)
        model.load_state_dict(torch.load(run / f"epoch{best}.pt", map_location=dev, weights_only=False)["model"])
        out[seed] = score_part(model, valid, tables, dev)
        np.save(cache, out[seed])
    return out


# ------------------------------------------------------------------ M7d late fusion
def user_halves(valid):
    users = valid.ctx.imps["user_id"].to_numpy()
    uniq = np.unique(users)
    half_a = set(uniq[np.random.default_rng(SEED).permutation(len(uniq))[: len(uniq) // 2]])
    return np.array([u in half_a for u in users])          # per impression


def per_imp_auc(truth, scores):
    return fast_metrics(truth["imp_key"].to_numpy(), truth["label"].to_numpy(), scores)["auc"].to_numpy()


def fusion_grid(valid, base, pm):
    """Per-impression AUC for every alpha. -> (n_imps, n_alphas)."""
    codes = valid.ctx.imp_codes
    b, e = minmax_within(codes, base), minmax_within(codes, pm)
    return np.column_stack([per_imp_auc(valid.ctx.truth, fuse(b, e, a)) for a in ALPHAS])


def cross_fit(grid, in_a):
    """Alpha chosen on one user half, applied to the other. -> (per-impression AUC, alpha used on A, alpha used on B)."""
    ia = int(np.nanargmax(np.nanmean(grid[~in_a], axis=0)))      # chosen on B, used on A
    ib = int(np.nanargmax(np.nanmean(grid[in_a], axis=0)))
    return np.where(in_a, grid[:, ia], grid[:, ib]), ALPHAS[ia], ALPHAS[ib]


def run_fusion():
    dev = torch.device("cuda")
    tokens, emb, emo, pct, fit, valid, fit_items = setup()
    tables = Tables(tokens, emo, pct, fit_items, dev)
    base = m6_seed_scores(valid, tables, emb, dev)
    in_a = user_halves(valid)
    perm = control_perm(len(valid.ctx.vectors))
    rows, grid_rows, fused = [], [], {"M7d": [], "M7d_ctrl": []}
    for name, vec in (("M7d", valid.ctx.vectors), ("M7d_ctrl", valid.ctx.vectors[perm])):
        prof, _ = window_profiles(valid.ctx.hist_rows, vec, 50, 0.9)
        pm = personal_match(valid.ctx.groups, valid.ctx.item_idx, prof, vec)
        for seed in SEEDS:
            g = fusion_grid(valid, base[seed], pm)
            for a, v in zip(ALPHAS, np.nanmean(g, axis=0)):
                grid_rows.append({"model": name, "seed": seed, "alpha": a, "valid_auc_all": v})
            cf, a_a, a_b = cross_fit(g, in_a)
            full_alpha = ALPHAS[int(np.nanargmax(np.nanmean(g, axis=0)))]
            codes = valid.ctx.imp_codes
            s = fuse(minmax_within(codes, base[seed]), minmax_within(codes, pm), full_alpha)
            fused[name].append(s)
            r = summary_row(name, seed, valid.ctx.truth, s, {"alpha_all_valid": full_alpha, "alpha_half_a": a_a, "alpha_half_b": a_b,
                                                             "auc_crossfit": float(np.nanmean(cf))})
            rows.append(r)
            print(name, seed, {k: r[k] for k in ("alpha_all_valid", "alpha_half_a", "alpha_half_b", "auc_crossfit", "auc")}, flush=True)
    for seed in SEEDS:
        rows.append(summary_row("M6", seed, valid.ctx.truth, base[seed], {"auc_crossfit": float(np.nanmean(per_imp_auc(valid.ctx.truth, base[seed])))}))
    add_seed_rows(rows)
    pd.DataFrame(grid_rows).to_csv(TABLES / "v8_m7d_alpha_grid.csv", index=False)
    for name in ("M7d", "M7d_ctrl"):
        save_mean(name, valid.ctx.truth, fused[name], {"alphas": ALPHAS, "per_seed_alpha": "best on all valid", "control": name.endswith("ctrl")})


# ------------------------------------------------------------------ M7e / M7f training
def run_train(models, control=False):
    dev = torch.device("cuda")
    perm = control_perm(len(news_tables()[0])) if control else None
    tokens, emb, emo, pct, fit, valid, fit_items = setup(perm)
    tables = Tables(tokens, emo, pct, fit_items, dev, perm=perm)
    tables.pm_mu, tables.pm_sd = personal_match_stats(fit, tables)
    print(f"personal match on fit: mean {tables.pm_mu:.5f}, SD {tables.pm_sd:.5f}", flush=True)
    rows = []
    for m in models:
        name = f"{m}_ctrl" if control else m
        scores = []
        for seed in SEEDS:
            model, hist = train_run(f"{name}_s{seed}", MODES[m], seed, fit, np.arange(len(fit.cands)), valid, np.arange(len(valid.cands)),
                                    tables, emb, dev, emo_lr=EMO_LR)
            s = score_part(model, valid, tables, dev)
            np.save(RUN_DIR / f"{name}_s{seed}" / "valid_scores.npy", s)
            scores.append(s)
            best = max(hist, key=lambda h: h["valid_auc"])
            w = {k: float(v.detach().cpu()) for k, v in model.named_parameters() if k in ("w0", "w1")}
            rows.append(summary_row(name, seed, valid.ctx.truth, s, {"best_epoch": best["epoch"], "epochs_run": len(hist), **w,
                                                                     "auc_crossfit": None}))
            pd.DataFrame(hist).assign(model=name, seed=seed).to_csv(RUN_DIR / f"{name}_s{seed}" / "curve.csv", index=False)
            print(rows[-1], flush=True)
            del model
            torch.cuda.empty_cache()
        add_seed_rows(rows[-len(SEEDS):])
        save_mean(name, valid.ctx.truth, scores, {"mode": MODES[m], "seeds": SEEDS, "emo_lr": EMO_LR, "pm_mu": tables.pm_mu,
                                                  "pm_sd": tables.pm_sd, "control": control})


# ------------------------------------------------------------------ selection (plan v8 = plan v6 Rule 8)
def select():
    t = pd.read_csv(SEEDS_TABLE)
    v6 = pd.read_csv(TABLES / "v6_nrms_seeds_valid.csv")
    noise = float(v6[v6["model"] == "M6"]["auc"].std())
    m6 = t[t["model"] == "M6"].set_index("seed")["auc"]
    rows = []
    for v in ORDER:
        s = t[t["model"] == v].set_index("seed")
        if s.empty:
            continue
        auc = s["auc_crossfit"] if v == "M7d" else s["auc"]          # M7d: the cross-fitted number decides
        d = auc - m6.loc[auc.index]
        ctrl = t[t["model"] == f"{v}_ctrl"]
        rows.append({"variant": v, "mean_auc": float(auc.mean()), "gain_vs_m6": float(d.mean()), "seed_wins": int((d > 0).sum()),
                     "m6_seed_sd": noise, "eligible": bool((d > 0).sum() >= 2 and d.mean() > noise),
                     "mean_ndcg@10": float(s["ndcg@10"].mean()), "m6_ndcg@10": float(t[t["model"] == "M6"]["ndcg@10"].mean()),
                     "ctrl_mean_auc": float(ctrl["auc"].mean()) if len(ctrl) else np.nan})
    r = pd.DataFrame(rows)
    elig = r[r["eligible"]]
    sel = {"rule": "plan v8 = plan v6 Rule 8 (seeds), D105", "m6_mean_auc": float(m6.mean()), "m6_seed_sd": noise,
           "passed_rule": bool(len(elig))}
    if len(elig):
        top = elig["mean_auc"].max()
        chosen = next(v for v in ORDER if v in set(elig["variant"]) and top - float(elig.set_index("variant").loc[v, "mean_auc"]) < 0.0005)
        c = r.set_index("variant").loc[chosen]
        sel.update({"selected": chosen, "selected_name": NAMES[chosen], "gain_vs_m6": float(c["gain_vs_m6"]),
                    "ctrl_mean_auc": None if np.isnan(c["ctrl_mean_auc"]) else float(c["ctrl_mean_auc"]),
                    "beats_ctrl": None if np.isnan(c["ctrl_mean_auc"]) else bool(c["mean_auc"] > c["ctrl_mean_auc"])})
    else:
        sel["selected"] = None
    r.to_csv(TABLES / "v8_nrms_rule8.csv", index=False)
    (TABLES / "v8_nrms_selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    print(r.round(5).to_string(index=False))
    print(json.dumps(sel, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", action="store_true")
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--ctrl", nargs="*", default=[])
    ap.add_argument("--select", action="store_true")
    a = ap.parse_args()
    if a.fusion:
        run_fusion()
    if a.train:
        run_train(a.train)
    if a.ctrl:
        run_train(a.ctrl, control=True)
    if a.select:
        select()


if __name__ == "__main__":
    main()
