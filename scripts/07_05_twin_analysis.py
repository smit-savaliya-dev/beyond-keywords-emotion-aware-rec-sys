"""Stage 6 (plan v6): where does each final emotion model differ from its no-emotion twin? M5v6 vs M4v6, M7 vs M6, B1 vs B0, on valid
(fold F4) and dev (read from the single saved prediction files): by consistency tertile, clicked category, history length, persona, time slot.

Reads:  saved per-impression files; behaviours; news; E1 cut-offs; spec v5; frozen settings. Writes: results/tables/v6_twin_subgroups.csv,
results/figures/v6_improve/twin_*.png. Not a pipeline step: it reads saved dev result files and chooses nothing.
Run:    python scripts/07_05_twin_analysis.py
"""
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import behaviour_charts as bc
from src.behaviour import assign_tertile, history_consistency, time_slot, window_profiles
from src.behaviour_runs import load_context
from src.config import BEHAVIOUR_SPEC_VERSION, FIGURES, FROZEN_DIR, PROCESSED, RESULTS, TABLES
from src.emotion_features import GO_EMOTIONS
from src.emotion_spec import load_spec
from src.ltr_v6 import persona_model
from src.run_registry import load_rows
from src.subgroups import group_gains
from src.where_it_helps import HISTORY_BINS, clicked_category, history_bin, top_categories

OUT = FIGURES / "v6_improve"
NRMS = {"M7a": "M7a_nrms_emotion_item", "M7b": "M7b_nrms_emotion_gated", "M7c": "M7c_nrms_emotion_both"}


def main():
    spec = load_spec(BEHAVIOUR_SPEC_VERSION)["behaviour"]
    m5src = json.loads((FROZEN_DIR / "M5v6.json").read_text(encoding="utf-8"))["config"]["source"]
    m7 = NRMS[json.loads((TABLES / "v6_nrms_selection.json").read_text(encoding="utf-8"))["selected"]]
    e1 = json.loads((TABLES / "behaviour_e1_meta.json").read_text(encoding="utf-8"))
    k = json.loads((TABLES / "behaviour_e8_meta.json").read_text(encoding="utf-8"))
    fit = load_context("fit", with_m2=False)
    km, mu, sd = persona_model(fit, k=k["k"])
    names = k["names"]
    del fit
    news = pd.read_parquet(PROCESSED / "news.parquet", columns=["news_id", "category"]).set_index("news_id")["category"].astype(str)
    fc = load_rows(PROCESSED, ["fit"], ["news_id", "label"])
    top = top_categories(fc.loc[fc["label"] == 1, "news_id"].astype(str).to_numpy(), news)
    rows = []
    for part in ("valid", "dev"):
        ctx = load_context(part, with_m2=False)
        idx, r = ctx.imps["imp_key"].to_numpy(), ctx.imps["prof_row"].to_numpy()
        pair, _ = history_consistency(ctx.hist_rows, ctx.vectors)
        prof, used = window_profiles(ctx.hist_rows, ctx.vectors, 50, 0.9)
        lab = np.where(used >= 5, km.predict((prof - mu) / sd), -1)
        splits = {"consistency tertile": pd.Series(assign_tertile(pair[r], e1["cuts_fit"]), index=idx),
                  "clicked category": clicked_category(ctx.truth, news, top),
                  "known history items": pd.Series(history_bin(used[r]), index=idx),
                  "persona": pd.Series([names[i] if i >= 0 else "too little history" for i in lab[r]], index=idx),
                  "time slot (exploratory)": pd.Series(time_slot(ctx.imps["time"], spec["time_slots"]), index=idx)}
        pairs = [("M4v6", m5src if part == "valid" else "M5v6", "M5v6 - M4v6"), ("M6_nrms", m7, "M7 - M6"), ("B0", "B1", "B1 - B0")]
        for a, b, label in pairs:
            pa, pb = (pd.read_parquet(RESULTS / "per_impression" / f"{m}__{part}.parquet") for m in (a, b))
            for split, groups in splits.items():
                rows.append(group_gains(pa, pb, groups, ctx.users_by_imp, metrics=("auc",), name_a=a, name_b=b).assign(part=part, pair=label, split=split))
        print(part, "done", flush=True)
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(TABLES / "v6_twin_subgroups.csv", index=False)
    d = t[t["part"] == "dev"]
    order = {"consistency tertile": ["low", "mid", "high", "no_history"], "persona": [*names, "too little history"],
             "known history items": [b[2] for b in HISTORY_BINS]}
    cols = {"M5v6 - M4v6": bc.BLUE, "M7 - M6": bc.ORANGE, "B1 - B0": bc.PINK}
    for split, fname, title in (("consistency tertile", "twin_gain_by_consistency.png", "Does emotion still help consistent users more?"),
                                ("persona", "twin_gain_by_persona.png", "Emotion against its twin, per reader persona"),
                                ("known history items", "twin_gain_by_history.png", "Emotion against its twin, by history length")):
        x = d[d["split"] == split].set_index(["pair", "group"])
        groups = order[split]
        bc.grouped(OUT / fname, title,
                   "Final-test AUC of each emotion model minus its no-emotion twin, per group (from the saved dev prediction files; a second look at dev), "
                   "95% user-bootstrap CIs.",
                   [g.split(": ")[0] if ": " in g else g for g in groups],
                   [(p, [x.loc[(p, g), "gain"] if (p, g) in x.index else np.nan for g in groups],
                     [x.loc[(p, g), "ci_low"] if (p, g) in x.index else np.nan for g in groups],
                     [x.loc[(p, g), "ci_high"] if (p, g) in x.index else np.nan for g in groups], c) for p, c in cols.items()],
                   "AUC gain over the twin (final test)", fmt="{:+.4f}", zero=True,
                   notes=[g.split(": ")[1] if ": " in g else "" for g in groups] if split == "persona" else None)


if __name__ == "__main__":
    main()
