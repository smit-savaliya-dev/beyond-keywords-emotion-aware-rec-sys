"""E6 (plan v5, exploratory): ranking quality and the emotion gain by time of day and weekday/weekend, and the emotion of clicked items by hour.
Fit scores are computed with the frozen M2 and M3 settings; valid and dev come from the saved prediction files. No claims, no family.

Reads:  fit, valid and dev behaviours and impressions; saved M2 and M3 per-impression files; the fit M4 feature table (M2 score); spec v5.
Writes: results/tables/behaviour_e6_*.csv; results/figures/v5_behaviour/e6_*.png.
Run:    python scripts/06_06_e6_time_patterns.py
"""
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import behaviour_charts as bc
from src.behaviour import time_slot
from src.behaviour_runs import emotion_score, fused_scores, load_context, metrics
from src.config import BEHAVIOUR_FIGS, BEHAVIOUR_SPEC_VERSION, M2_NAME, M3_NAME, RESULTS, TABLES
from src.emotion_features import GO_EMOTIONS
from src.emotion_spec import load_spec
from src.subgroups import group_gains

SLOTS_ORDER = ["night", "morning", "afternoon", "evening"]


def main():
    slots = load_spec(BEHAVIOUR_SPEC_VERSION)["behaviour"]["time_slots"]
    gains, clicked = [], []
    for part in ("fit", "valid", "dev"):
        ctx = load_context(part, with_m2=(part == "fit"))
        users = ctx.users_by_imp
        if part == "fit":
            p2, p3 = metrics(ctx, ctx.m2), metrics(ctx, fused_scores(ctx, emotion_score(ctx), 0.8))
        else:
            p2, p3 = (pd.read_parquet(RESULTS / "per_impression" / f"{m}__{part}.parquet") for m in (M2_NAME, M3_NAME))
        slot = pd.Series(time_slot(ctx.imps["time"], slots), index=ctx.imps["imp_key"].to_numpy())
        gains.append(group_gains(p2, p3, slot, users, metrics=("auc",), name_a="M2", name_b="M3").assign(part=part, split="time of day"))
        if part == "fit":
            wk = pd.Series(np.where(ctx.imps["time"].dt.dayofweek.to_numpy() >= 5, "weekend", "weekday"), index=ctx.imps["imp_key"].to_numpy())
            gains.append(group_gains(p2, p3, wk, users, metrics=("auc",), name_a="M2", name_b="M3").assign(part=part, split="weekday or weekend"))
        if part in ("fit", "valid"):
            y = ctx.truth["label"].to_numpy() == 1
            hours = ctx.imps["time"].dt.hour.to_numpy()[ctx.imp_codes[y]]
            clicked.append(pd.DataFrame(ctx.vectors[ctx.item_idx[y]], columns=GO_EMOTIONS).assign(hour=hours, part=part))
        print(part, "done", flush=True)
    gains = pd.concat(gains, ignore_index=True)
    gains.to_csv(TABLES / "behaviour_e6_gain_by_time.csv", index=False)
    c = pd.concat(clicked, ignore_index=True)
    by_hour = c.groupby("hour")[GO_EMOTIONS].mean()
    centred = by_hour - c[GO_EMOTIONS].mean()
    out = by_hour.assign(clicks=c.groupby("hour").size()).reset_index()
    out.to_csv(TABLES / "behaviour_e6_clicked_emotion_by_hour.csv", index=False)
    print(gains.round(5).to_string(index=False))

    g = gains[gains["split"] == "time of day"].set_index(["part", "group"])
    bc.grouped(BEHAVIOUR_FIGS / "e6_gain_by_time_slot.png", "Does the emotion gain depend on the time of day? (exploratory)",
               "AUC gain of M3 over M2 by impression time slot (night 0-5, morning 6-11, afternoon 12-17, evening 18-23), 95% user-bootstrap CIs. "
               "Exploratory: one week of data, no correction for many looks.",
               SLOTS_ORDER, [(p, [g.loc[(p, s), "gain"] for s in SLOTS_ORDER], [g.loc[(p, s), "ci_low"] for s in SLOTS_ORDER],
                              [g.loc[(p, s), "ci_high"] for s in SLOTS_ORDER], col)
                             for p, col in (("fit", bc.SKY), ("valid", bc.BLUE), ("dev", bc.ORANGE))],
               "AUC gain (M3 minus M2)", fmt="{:+.3f}", zero=True,
               notes=[f"dev n = {int(g.loc[('dev', s), 'impressions']):,}" for s in SLOTS_ORDER])
    bc.heatmap(BEHAVIOUR_FIGS / "e6_clicked_emotion_by_hour.png", "Which emotions do people click at which hour? (exploratory)",
               "Mean emotion percentile of clicked items per hour minus the overall mean (fit and valid clicks). Orange = more than usual, blue = less.",
               (centred.T * 100).to_numpy(), GO_EMOTIONS, [str(h) for h in by_hour.index], "Hour of the day", "Emotion of the clicked item",
               fmt="{:.1f}", center=0.0)


if __name__ == "__main__":
    main()
