"""E9 (plan v5): does emotion matching narrow what users see? Emotion diversity (distinct v2b labels) and category diversity (distinct
categories) of each model's top 10 per impression, valid and dev, from the saved prediction files. Paired t-tests against M2. Descriptive.

Reads:  saved scores of M2, M3, M3b+, M4, M5b; news categories; v2b labels; impressions.
Writes: results/tables/behaviour_e9_diversity.csv; results/figures/v5_behaviour/e9_diversity.png.
Not a pipeline step: it reads saved dev result files.
Run:    python scripts/06_11_e9_diversity.py
"""
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import behaviour_charts as bc
from src.behaviour import top_k_distinct
from src.behaviour_runs import load_context
from src.config import BEHAVIOUR_FIGS, BEHAVIOUR_SPEC_VERSION, M2_NAME, M3_NAME, M3BP_NAME, M4_NAME, M5B_NAME, PROCESSED, RESULTS, SEED, TABLES
from src.emotion_spec import load_spec
from src.stats import paired_test

MODELS = {M2_NAME: "M2 content only", M3_NAME: "M3 + emotion", M3BP_NAME: "M3b+ adaptive emotion", M4_NAME: "M4 LightGBM", M5B_NAME: "M5b LightGBM + emotion"}


def main():
    k = load_spec(BEHAVIOUR_SPEC_VERSION)["behaviour"]["diversity_top_k"]
    news = pd.read_parquet(PROCESSED / "news.parquet", columns=["news_id", "category"]).set_index("news_id")["category"].astype(str)
    lab = pd.read_parquet(PROCESSED / "news_emotion_labels_variants.parquet", columns=["news_id", "label_v2b"]).set_index("news_id")["label_v2b"].astype(str)
    rows = []
    for part in ("valid", "dev"):
        ctx = load_context(part, with_m2=False)
        ids = ctx.truth["news_id"].to_numpy()
        cats, emos = news.reindex(ids).to_numpy(), lab.reindex(ids).to_numpy()
        keys = pd.MultiIndex.from_frame(ctx.truth[["imp_key", "news_id"]])
        div = {}
        for m in MODELS:
            s = pd.read_parquet(RESULTS / "scores" / f"{m}__{part}.parquet").set_index(["imp_key", "news_id"])["score"].reindex(keys).to_numpy()
            div[m] = {"emotion": top_k_distinct(ctx.imp_codes, s, emos, k), "category": top_k_distinct(ctx.imp_codes, s, cats, k)}
        for m in MODELS:
            for kind in ("emotion", "category"):
                t = paired_test(div[M2_NAME][kind], div[m][kind])
                rows.append({"part": part, "model": m, "label": MODELS[m], "diversity": kind, "mean": float(np.mean(div[m][kind])),
                             "diff_vs_m2": t["mean_diff"], "ci_low": t["ci_low"], "ci_high": t["ci_high"], "dz": t["dz"], "p": t["p"]})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "behaviour_e9_diversity.csv", index=False)
    print(t.round(4).to_string(index=False))
    d = t[t["part"] == "dev"].set_index(["model", "diversity"])
    e3 = d.loc[(M3BP_NAME, "emotion")]
    bc.grouped(BEHAVIOUR_FIGS / "e9_diversity.png", "Does emotion matching narrow what people see?",
               f"Distinct emotions and distinct categories among each model's top {k} per impression (final test). M3b+ vs M2: emotion diversity "
               f"{e3['diff_vs_m2']:+.3f}. Ethics note: a ranking that narrows emotional variety can reinforce mood loops; we only measure it here.",
               list(MODELS.values()), [(kind, [d.loc[(m, kind), "mean"] for m in MODELS], None, None, col)
                                       for kind, col in (("emotions in top 10", bc.ORANGE), ("categories in top 10", bc.BLUE))],
               "Distinct values in the top 10 (final test)", fmt="{:.2f}")


if __name__ == "__main__":
    main()
