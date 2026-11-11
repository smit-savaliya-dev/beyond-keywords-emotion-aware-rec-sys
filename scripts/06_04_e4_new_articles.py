"""E4 (plan v5): new articles. Exposure of a candidate = strictly earlier impressions that showed the item (any part, by timestamp).
Buckets new / few / many with the cut fixed on fit. AUC within each bucket for M0 to M5 from the saved valid and dev score files.

Reads:  impressions (all parts: item and time only for exposure, labels of valid and dev for the bucket AUC), saved scores of M0 to M5.
Writes: results/tables/behaviour_e4_*.csv, behaviour_e4_meta.json; archive/figures/v5_behaviour/e4_*.png.
Not a pipeline step: it reads saved dev result files. Nothing is chosen here.
Run:    python scripts/06_04_e4_new_articles.py
"""
import json

import numpy as np
import pandas as pd
from scipy import stats as st

import _bootstrap  # noqa: F401
from src import behaviour_charts as bc
from src.behaviour import exposure_bucket
from src.behaviour_runs import exposure_table
from src.click_signal import permute_within_groups
from src.config import BEHAVIOUR_FIGS, M0_NAME, M1_NAME, M2_NAME, M3_NAME, M4_NAME, M5_NAME, PROCESSED, RESULTS, SEED, TABLES
from src.evaluate import load_truth
from src.provenance import run_info
from src.stats import holm
from src.subgroups import group_gains, group_means, restricted_metrics, shuffled_labels

MODELS = {M0_NAME: "M0 popularity", M1_NAME: "M1 TF-IDF", M2_NAME: "M2 Sentence-BERT", M3_NAME: "M3 = M2 + emotion",
          M4_NAME: "M4 LightGBM", M5_NAME: "M5 = M4 + emotion"}
BUCKETS = ["new", "few", "many"]


def main():
    expo, q = exposure_table(PROCESSED)
    print(f"Q (median fit exposure > 0) = {q:.0f}")
    gains, means, tests, controls, shares = [], [], [], [], []
    for part in ("valid", "dev"):
        truth = load_truth(part, PROCESSED)
        keys = pd.MultiIndex.from_frame(truth[["imp_key", "news_id"]])
        bucket = exposure_bucket(expo.reindex(keys).to_numpy(), q)
        beh = pd.read_parquet(PROCESSED / "behaviors.parquet", columns=["imp_key", "part", "user_id"], filters=[("part", "==", part)])
        users = beh.set_index(beh["imp_key"].astype(str))["user_id"].astype(str)
        for b in BUCKETS:
            shares.append({"part": part, "bucket": b, "share_rows": float((bucket == b).mean()),
                           "share_clicks": float((bucket[truth["label"].to_numpy() == 1] == b).mean())})
        scores = {m: pd.read_parquet(RESULTS / "scores" / f"{m}__{part}.parquet").set_index(["imp_key", "news_id"])["score"].reindex(keys).to_numpy()
                  for m in MODELS}
        sh = shuffled_labels(truth, SEED)
        perm_bucket = permute_within_groups(pd.factorize(truth["imp_key"])[0], bucket, SEED)   # buckets shuffled within impressions
        per = {}
        for b in BUCKETS:
            mask = bucket == b
            for m in MODELS:
                per[(m, b)] = restricted_metrics(truth, scores[m], mask)
                groups = pd.Series(b, index=per[(m, b)]["imp_key"])
                means.append(group_means(per[(m, b)], groups, users, metrics=("auc",), name=MODELS[m]).assign(part=part, model_id=m))
            for a, c in ((M2_NAME, M3_NAME), (M4_NAME, M5_NAME)):
                groups = pd.Series(b, index=per[(a, b)]["imp_key"])
                gains.append(group_gains(per[(a, b)], per[(c, b)], groups, users, metrics=("auc",), name_a=a, name_b=c).assign(part=part))
        for a, c in ((M2_NAME, M3_NAME), (M4_NAME, M5_NAME)):
            d = {b: (per[(c, b)]["auc"] - per[(a, b)]["auc"]).dropna().to_numpy() for b in ("new", "many")}
            t = st.ttest_ind(d["new"], d["many"], equal_var=False)
            tests.append({"part": part, "pair": f"{c} - {a}", "gain_new": d["new"].mean(), "gain_many": d["many"].mean(),
                          "new_minus_many": d["new"].mean() - d["many"].mean(), "p": float(t.pvalue), "n_new": len(d["new"]), "n_many": len(d["many"])})
            for name, msk, labels in (("buckets shuffled within impressions", perm_bucket, truth["label"]), ("labels shuffled", bucket, sh["label"])):
                t2 = truth.assign(label=np.asarray(labels))
                dd = {b: (restricted_metrics(t2, scores[c], msk == b)["auc"] - restricted_metrics(t2, scores[a], msk == b)["auc"]).dropna().to_numpy()
                      for b in ("new", "many")}
                controls.append({"part": part, "pair": f"{c} - {a}", "control": name, "new_minus_many": dd["new"].mean() - dd["many"].mean(),
                                 "p": float(st.ttest_ind(dd["new"], dd["many"], equal_var=False).pvalue)})
        print(part, "done", flush=True)
    tests = pd.DataFrame(tests)
    tests["p_holm"] = tests.groupby("part")["p"].transform(lambda p: holm(p.to_numpy()))
    gains, means = pd.concat(gains, ignore_index=True), pd.concat(means, ignore_index=True)
    gains.to_csv(TABLES / "behaviour_e4_gain_by_bucket.csv", index=False)
    means.to_csv(TABLES / "behaviour_e4_auc_by_bucket.csv", index=False)
    tests.to_csv(TABLES / "behaviour_e4_tests.csv", index=False)
    pd.DataFrame(controls).to_csv(TABLES / "behaviour_e4_controls.csv", index=False)
    pd.DataFrame(shares).to_csv(TABLES / "behaviour_e4_bucket_shares.csv", index=False)
    (TABLES / "behaviour_e4_meta.json").write_text(json.dumps({"q_fit": q, "run_info": run_info()}, indent=2), encoding="utf-8")
    print(gains.round(5).to_string(index=False), tests.round(5).to_string(index=False), sep="\n")

    m = means[means["part"] == "dev"].set_index(["model_id", "group"])
    cols = [bc.SKY, bc.GREY, bc.BLUE, bc.ORANGE, bc.PURPLE, bc.YELLOW]
    sizes = [int(m.loc[(M2_NAME, b), "impressions"]) for b in BUCKETS]
    bc.grouped(BEHAVIOUR_FIGS / "e4_auc_by_exposure.png", "How well each model ranks new and familiar articles",
               f"Final-test AUC among the candidates of one exposure bucket (new = never shown before; few = shown 1-{q:.0f} times; many = more), "
               "from the saved prediction files; 95% user-bootstrap CIs.",
               ["new", f"few (1-{q:.0f})", f"many (>{q:.0f})"], [(MODELS[k], [m.loc[(k, b), "mean"] for b in BUCKETS],
                                                       [m.loc[(k, b), "ci_low"] for b in BUCKETS], [m.loc[(k, b), "ci_high"] for b in BUCKETS], col)
                                                      for k, col in zip(MODELS, cols)],
               "AUC within the bucket (final test)", fmt="{:.2f}", ylim=(0.4, None), notes=[f"n = {s:,}" for s in sizes], legend_loc="lower right")
    g = gains[(gains["model_a"] == M2_NAME)].set_index(["part", "group"])
    t = tests[(tests["pair"] == f"{M3_NAME} - {M2_NAME}") & (tests["part"] == "dev")].iloc[0]
    bc.grouped(BEHAVIOUR_FIGS / "e4_m3_gain_by_exposure.png", "Does emotion help most for new articles?",
               f"AUC gain of M3 over M2 within each exposure bucket, valid and final test, 95% user-bootstrap CIs. "
               f"Final test: new minus many {t['new_minus_many']:+.4f} (Holm p {t['p_holm']:.3g}).",
               BUCKETS, [(p, [g.loc[(p, b), "gain"] for b in BUCKETS], [g.loc[(p, b), "ci_low"] for b in BUCKETS], [g.loc[(p, b), "ci_high"] for b in BUCKETS], col)
                         for p, col in (("valid", bc.BLUE), ("dev", bc.ORANGE))],
               "AUC gain (M3 minus M2)", fmt="{:+.4f}", zero=True)


if __name__ == "__main__":
    main()
