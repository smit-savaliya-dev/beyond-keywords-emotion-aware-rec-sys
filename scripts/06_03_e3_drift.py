"""E3 (plan v5): emotion drift. Does a short emotion profile matter more for users whose recent reading differs from their older reading?
Drift tertiles fixed on fit; short (last 5) vs long (last 50, decay 0.9) profile on valid. The dev part (M3b+ minus M3 by drift tertile)
is in 06_09, read from the final prediction file.

Reads:  fit and valid behaviours and impressions, saved valid M2 scores, spec v5.
Writes: results/tables/behaviour_e3_*.csv, behaviour_e3_meta.json; archive/figures/v5_behaviour/e3_*.png.
Run:    python scripts/06_03_e3_drift.py
"""
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import behaviour_charts as bc
from src.behaviour import assign_tertile, drift_scores, tertile_cuts
from src.behaviour_runs import emotion_score, fused_scores, load_context, metrics
from src.config import BEHAVIOUR_FIGS, BEHAVIOUR_SPEC_VERSION, SEED, TABLES
from src.emotion_spec import load_spec
from src.provenance import run_info
from src.stats import holm
from src.subgroups import group_gains, shuffled_labels, trend_tests

ORDER = ["low", "mid", "high", "under_10_items"]


def drift(ctx, spec):
    d = drift_scores(ctx.hist_rows, ctx.vectors, spec["recent"], spec["older_max"], spec["min_items"])
    return d[ctx.imps["prof_row"].to_numpy()]


def main():
    spec = load_spec(BEHAVIOUR_SPEC_VERSION)["behaviour"]["drift"]
    fit = load_context("fit", with_m2=False)
    d_fit = drift(fit, spec)
    cuts = tertile_cuts(d_fit)
    counts, edges = np.histogram(d_fit[np.isfinite(d_fit)], bins=50)
    pd.DataFrame({"bin_low": edges[:-1], "bin_high": edges[1:], "impressions": counts}).to_csv(TABLES / "behaviour_e3_drift_fit.csv", index=False)
    del fit
    ctx = load_context("valid")
    d = drift(ctx, spec)
    groups = assign_tertile(d, cuts, missing="under_10_items")
    g = pd.Series(groups, index=ctx.imps["imp_key"].to_numpy())
    users = ctx.users_by_imp
    sw, lw = spec["short_window"], spec["long_window"]
    short = metrics(ctx, fused_scores(ctx, emotion_score(ctx, sw["size"], sw["decay"]), 0.8))
    long_ = metrics(ctx, fused_scores(ctx, emotion_score(ctx, lw["size"], lw["decay"]), 0.8))
    gains = group_gains(long_, short, g, users, name_a="long profile", name_b="short profile")
    gain = short["auc"].to_numpy() - long_["auc"].to_numpy()
    tr = trend_tests(d, gain, groups)
    tr["p_holm_rho"], tr["p_holm_hl"] = holm([tr["rho_p"], tr["hl_p"]])
    perm = np.random.default_rng(SEED).permutation(len(d))
    sh = shuffled_labels(ctx.truth, SEED)["label"].to_numpy()
    gsh = (metrics(ctx, fused_scores(ctx, emotion_score(ctx, sw["size"], sw["decay"]), 0.8), sh)["auc"].to_numpy()
           - metrics(ctx, fused_scores(ctx, emotion_score(ctx, lw["size"], lw["decay"]), 0.8), sh)["auc"].to_numpy())
    controls = pd.DataFrame([{"control": "drift permuted", **trend_tests(d[perm], gain, groups[perm])},
                             {"control": "labels shuffled", **trend_tests(d, gsh, groups)}])
    gains.to_csv(TABLES / "behaviour_e3_short_vs_long_by_drift.csv", index=False)
    pd.DataFrame([{"part": "valid", **tr}]).to_csv(TABLES / "behaviour_e3_trend.csv", index=False)
    controls.to_csv(TABLES / "behaviour_e3_controls.csv", index=False)
    (TABLES / "behaviour_e3_meta.json").write_text(json.dumps({"cuts_fit": cuts, "run_info": run_info()}, indent=2), encoding="utf-8")
    print(gains[gains["metric"] == "auc"].round(5).to_string(index=False), tr, controls.round(4).to_string(index=False), sep="\n")

    hist = pd.read_csv(TABLES / "behaviour_e3_drift_fit.csv")
    vals = np.repeat((hist["bin_low"] + hist["bin_high"]).to_numpy() / 2, hist["impressions"].to_numpy())
    bc.histogram(BEHAVIOUR_FIGS / "e3_drift_hist.png", "How much does a user's recent emotional reading drift?",
                 f"Drift = agreement among the last 5 history items minus their similarity to the older items (fit impressions, users with 10+ items). "
                 f"Dashed: tertile cut-offs {cuts[0]:.3f} and {cuts[1]:.3f}, fixed on fit.", vals, "Drift score", cuts=list(cuts),
                 cut_labels=["low | mid", "mid | high"])
    a = gains[gains["metric"] == "auc"].set_index("group").reindex(ORDER)
    bc.bars(BEHAVIOUR_FIGS / "e3_short_vs_long_by_drift.png", "Does a short emotion profile help users who drift?",
            f"Valid AUC of the last-5 emotion profile minus the last-50 profile (both fused with M2 at alpha 0.8), per drift tertile, 95% user-bootstrap CIs. "
            f"High minus low {tr['high_minus_low']:+.4f} (Holm p {tr['p_holm_hl']:.3g}).",
            [o.replace("_", " ") for o in ORDER], a["gain"].tolist(), "AUC: short minus long profile (valid)", low=a["ci_low"].tolist(),
            high=a["ci_high"].tolist(), colors=[bc.ORANGE if v > 0 else bc.GREY for v in a["gain"]],
            notes=[f"n = {int(n):,}" + (" (too small)" if s else "") for n, s in zip(a["impressions"], a["too_small"])])


if __name__ == "__main__":
    main()
