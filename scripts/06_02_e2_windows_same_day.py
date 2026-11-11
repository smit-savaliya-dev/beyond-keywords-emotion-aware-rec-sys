"""E2 (plan v5): short-window and same-day emotion profiles. The history window of the emotion profile and the same-day blend weight are
chosen on valid; M2 with the same windows is reported for context. Dev numbers come only from the final M3b+ file (06_07).

Reads:  fit and valid behaviours and impressions, the saved valid M2 scores, the M2 embeddings, spec v5.
Writes: results/tables/behaviour_e2_*.csv, behaviour_e2_choice.json; archive/figures/v5_behaviour/e2_*.png.
Run:    python scripts/06_02_e2_windows_same_day.py
"""
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import behaviour_charts as bc
from src.behaviour_runs import emotion_score, fused_scores, load_context, metrics, same_day
from src.config import BEHAVIOUR_FIGS, BEHAVIOUR_SPEC_VERSION, M2_NAME, PROCESSED, SEED, TABLES
from src.congruence import user_profiles
from src.emotion_spec import load_spec
from src.provenance import run_info
from src.run_registry import load_frozen
from src.sbert import load_embeddings, profile_cosine
from src.stats import holm, paired_test
from src.subgroups import paired_family, shuffled_labels


def m2_window(ctx, emb, size, decay):
    """M2 with a content profile of the newest `size` known history items; cold start scores 0 as in M2."""
    prof, _ = user_profiles(ctx.hist_rows, emb, np.zeros(emb.shape[1]), decay=decay, max_items=size)
    return profile_cosine(prof, emb, ctx.groups, ctx.item_idx)


def main():
    spec = load_spec(BEHAVIOUR_SPEC_VERSION)["behaviour"]
    w, sd_spec = spec["windows"], spec["same_day"]
    ctx = load_context("valid")
    users = ctx.users_by_imp
    m2 = metrics(ctx, ctx.m2)
    rows, per = [], {}
    emb = load_embeddings(load_frozen(M2_NAME)["config"]["text"], ctx.news_ids, PROCESSED)
    for size in w["sizes"]:
        for decay in w["decays"]:
            emo = emotion_score(ctx, size, decay)
            per[(size, decay)] = metrics(ctx, fused_scores(ctx, emo, w["alpha"]))
            t = paired_test(m2["auc"], per[(size, decay)]["auc"])
            c2 = metrics(ctx, m2_window(ctx, emb, size, decay))
            rows.append({"size": size, "decay": decay, "m3_auc": float(per[(size, decay)]["auc"].mean()), "gain_vs_m2": t["mean_diff"],
                         "ci_low": t["ci_low"], "ci_high": t["ci_high"], "m2_window_auc": float(c2["auc"].mean())})
            print(rows[-1], flush=True)
    del emb
    win = pd.DataFrame(rows)
    cur = (w["current"]["size"], w["current"]["decay"])
    best_row = win.loc[win["m3_auc"].idxmax()]
    best = (int(best_row["size"]), float(best_row["decay"]))
    if best_row["m3_auc"] - win.set_index(["size", "decay"]).loc[cur, "m3_auc"] < w["tie_tolerance"]:
        best = cur
    win.to_csv(TABLES / "behaviour_e2_windows.csv", index=False)

    sd, used, _ = same_day(ctx)
    covered = used > 0
    fit = load_context("fit", with_m2=False)
    _, used_fit, _ = same_day(fit)
    cov = pd.DataFrame([{"part": p, "impressions": len(u), "covered": int((u > 0).sum()), "share_covered": float((u > 0).mean()),
                         "median_earlier_clicks": float(np.median(u[u > 0])) if (u > 0).any() else 0.0} for p, u in (("fit", used_fit), ("valid", used))])
    cov.to_csv(TABLES / "behaviour_e2_coverage.csv", index=False)
    del fit
    blend, per_a = [], {}
    for a in sd_spec["blend_grid"]:
        per_a[a] = metrics(ctx, fused_scores(ctx, emotion_score(ctx, *best, sd, a), w["alpha"]))
        auc = per_a[a]["auc"].to_numpy()
        blend.append({"a": a, "valid_auc": float(np.nanmean(auc)), "covered_auc": float(np.nanmean(auc[covered])),
                      "uncovered_auc": float(np.nanmean(auc[~covered])), "gain_vs_m2": float(np.nanmean(auc - m2["auc"].to_numpy())),
                      "covered_gain_vs_m2": float(np.nanmean((auc - m2["auc"].to_numpy())[covered]))})
    blend = pd.DataFrame(blend)
    a_best = float(blend.loc[blend["valid_auc"].idxmax(), "a"])
    cov_test = paired_test(per_a[1.0]["auc"].to_numpy()[covered], per_a[a_best]["auc"].to_numpy()[covered])
    keep_blend = a_best < 1.0 and cov_test["mean_diff"] > 0 and cov_test["p"] < 0.05
    blend.to_csv(TABLES / "behaviour_e2_blend.csv", index=False)

    fam = paired_family([("best window - current window", per[cur], per[best])], users)
    t = paired_test(per_a[1.0]["auc"].to_numpy()[covered], per_a[a_best]["auc"].to_numpy()[covered])
    fam = pd.concat([fam, pd.DataFrame([{"comparison": f"blend a={a_best} - a=1, covered impressions", "metric": "auc",
                                         **{k: t[k] for k in ("n", "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high", "dz", "p")}}])], ignore_index=True)
    fam["p_holm"] = holm(fam["p"].to_numpy())
    fam["significant_holm"] = fam["p_holm"] < 0.05
    a_final = a_best if keep_blend else 1.0
    sh = shuffled_labels(ctx.truth, SEED)["label"].to_numpy()
    ctrl = paired_family([("chosen profile - M2, labels shuffled", metrics(ctx, ctx.m2, sh),
                           metrics(ctx, fused_scores(ctx, emotion_score(ctx, *best, sd, a_final), w["alpha"]), sh))], users)
    fam = pd.concat([fam, ctrl.assign(family="control")], ignore_index=True)
    fam.to_csv(TABLES / "behaviour_e2_family.csv", index=False)
    split = []
    for name, p in (("current M3 profile", per[cur]), (f"chosen (window {best[0]}, decay {best[1]}, a {a_final})", per_a[a_final])):
        for label, mask in (("covered", covered), ("not covered", ~covered)):
            tt = paired_test(m2["auc"].to_numpy()[mask], p["auc"].to_numpy()[mask])
            split.append({"profile": name, "impressions": label, "n": tt["n"], "gain_vs_m2": tt["mean_diff"], "ci_low": tt["ci_low"], "ci_high": tt["ci_high"], "p": tt["p"]})
    split = pd.DataFrame(split)
    split.to_csv(TABLES / "behaviour_e2_covered_split.csv", index=False)
    choice = {"window": {"size": best[0], "decay": best[1]}, "blend_a": a_final, "blend_a_best": a_best, "blend_kept": bool(keep_blend),
              "covered_test": cov_test, "coverage_valid": float(covered.mean()), "run_info": run_info()}
    (TABLES / "behaviour_e2_choice.json").write_text(json.dumps(choice, indent=2, default=float), encoding="utf-8")
    print(win.round(5).to_string(index=False), blend.round(5).to_string(index=False), fam.round(5).to_string(index=False), sep="\n\n")

    # charts
    x = w["sizes"]
    series = [(f"decay {d}" if d < 1 else "no decay", [win.set_index(["size", "decay"]).loc[(s, d), "gain_vs_m2"] for s in x],
               [win.set_index(["size", "decay"]).loc[(s, d), "ci_low"] for s in x], [win.set_index(["size", "decay"]).loc[(s, d), "ci_high"] for s in x],
               col, mk) for d, col, mk in ((0.9, bc.ORANGE, "o"), (1.0, bc.BLUE, "s"))]
    bc.lines(BEHAVIOUR_FIGS / "e2_window_m3_gain.png", "Does a shorter emotional history help?",
             f"Valid AUC gain of M3-style fusion over M2 by the number of history items in the emotion profile (alpha 0.8, shaded 95% CI). "
             f"Chosen: last {best[0]}, {'decay 0.9' if best[1] < 1 else 'no decay'}. Valid chose it, so it is optimistic.",
             x, series, "History items in the emotion profile", "AUC gain over M2 (valid)", fmt="{:+.4f}", xticks=[str(s) for s in x])
    series = [(f"decay {d}" if d < 1 else "no decay (M2 as frozen at 50)", [win.set_index(["size", "decay"]).loc[(s, d), "m2_window_auc"] for s in x],
               None, None, col, mk) for d, col, mk in ((0.9, bc.SKY, "o"), (1.0, bc.BLUE, "s"))]
    bc.lines(BEHAVIOUR_FIGS / "e2_window_m2_auc.png", "Content-only ranking with shorter histories",
             "Valid AUC of M2 (Sentence-BERT, no emotion) when its content profile uses only the newest history items. Reported for context; M2 is not changed.",
             x, series, "History items in the content profile", "AUC (valid)", fmt="{:.4f}", xticks=[str(s) for s in x])
    bc.bars(BEHAVIOUR_FIGS / "e2_same_day_coverage.png", "How often do we know what a user read earlier that day?",
            "Share of impressions with at least one click in an earlier impression on the same day, which the same-day profile needs.",
            ["fit (9-13 Nov)", "valid (14 Nov)"], [100 * v for v in cov["share_covered"]], "Impressions covered (%)", fmt="{:.1f}%",
            colors=[bc.BLUE, bc.ORANGE], zero=False)
    bc.lines(BEHAVIOUR_FIGS / "e2_blend_gain.png", "How much weight should the same-day profile get?",
             f"Valid AUC gain over M2 when the emotion profile is a * long-term + (1 - a) * same-day (a = 1: long-term only). Best a = {a_best}; "
             f"{'kept' if keep_blend else 'not kept'} by the value rule (covered gain {cov_test['mean_diff']:+.4f}, p {cov_test['p']:.3g}).",
             sd_spec["blend_grid"], [("all impressions", blend["gain_vs_m2"].tolist(), None, None, bc.BLUE, "o"),
                                     ("covered impressions", blend["covered_gain_vs_m2"].tolist(), None, None, bc.ORANGE, "s")],
             "a (weight of the long-term profile)", "AUC gain over M2 (valid)", fmt="{:+.4f}")
    s = split.set_index(["profile", "impressions"])
    names = list(dict.fromkeys(split["profile"]))
    bc.grouped(BEHAVIOUR_FIGS / "e2_covered_vs_uncovered.png", "Gain with and without same-day context",
               "Valid AUC gain over M2 for impressions with an earlier same-day click (covered) and without, for the current M3 profile and the chosen one.",
               ["covered", "not covered"], [(n, [s.loc[(n, g), "gain_vs_m2"] for g in ("covered", "not covered")],
                                             [s.loc[(n, g), "ci_low"] for g in ("covered", "not covered")],
                                             [s.loc[(n, g), "ci_high"] for g in ("covered", "not covered")], col)
                                            for n, col in zip(names, (bc.BLUE, bc.ORANGE))],
               "AUC gain over M2 (valid)", fmt="{:+.4f}", zero=True,
               notes=[f"n = {int(s.loc[(names[0], g), 'n']):,}" for g in ("covered", "not covered")])


if __name__ == "__main__":
    main()
