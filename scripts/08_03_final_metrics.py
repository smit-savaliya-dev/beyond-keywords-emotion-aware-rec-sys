"""Final multi-metric evaluation (plan v7 follow-up): every final model, many metrics, from the SAVED final dev predictions.

No model is trained and no model is newly scored on dev: the inputs are the saved score files in results/scores/. The only new score
vectors are the two same-weight Combined + emotion blends, rebuilt from the saved component scores at the Combined twin's own
valid-chosen weight (the fair comparison, D97). Metrics use src.metrics through src.final_eval, with the tie-breaks of src.evaluate,
and are checked against the saved per-impression files.

Reads:  results/scores/*__dev.parquet, results/per_impression/*__dev.parquet, results/tables/v6_blends_choice.json,
        v7_blends_choice.json; data/processed impressions, news (categories), news_emotion_labels_variants (v2b labels).
Writes: results/tables/final_all_metrics.csv, final_twin_gains_all_metrics.csv, final_dataset_overview.csv.
Run:    python scripts/08_03_final_metrics.py
"""
import importlib
import json
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import PROCESSED, RESULTS, TABLES
from src.evaluate import load_truth
from src.final_eval import PER_IMP, RANKING, evaluate_with_lists, load_users, twin_gains

# code -> (display name, family, setting, with emotion)
STD, RT = "Standard", "Real-time"
FINAL = {
    "M0_popularity": ("Popularity", "Baselines", STD, False),
    "M1_tfidf": ("Keyword match (TF-IDF)", "Baselines", STD, False),
    "M2_sbert": ("Meaning match (Sentence-BERT)", "Meaning", STD, False),
    "M3_sbert_emotion": ("Meaning + emotion", "Meaning", STD, True),
    "M3b_plus": ("Meaning + emotion, tuned profile", "Meaning", STD, True),
    "M4v6_R1": ("LightGBM ranker", "LightGBM", STD, False),
    "M5v6_R1": ("LightGBM + emotion", "LightGBM", STD, True),
    "M6_nrms": ("Neural ranker (NRMS)", "Neural", STD, False),
    "M7b_nrms_emotion_gated": ("Neural + emotion", "Neural", STD, True),
    "B0_R1": ("Combined ranker (LightGBM + neural)", "Combined", STD, False),
    "B1_R1": ("Combined + emotion", "Combined", STD, True),
    "B1_R1_same_weight": ("Combined + emotion (at the twin's weight)", "Combined", STD, True),
    "M4v6": ("LightGBM ranker", "LightGBM", RT, False),
    "M5v6": ("LightGBM + emotion", "LightGBM", RT, True),
    "B0": ("Combined ranker (LightGBM + neural)", "Combined", RT, False),
    "B1": ("Combined + emotion", "Combined", RT, True),
    "B1_same_weight": ("Combined + emotion (at the twin's weight)", "Combined", RT, True),
}
PAIRS = [("Meaning + emotion vs Meaning match", "M2_sbert", "M3_sbert_emotion"),
         ("Tuned profile vs Meaning match", "M2_sbert", "M3b_plus"),
         ("LightGBM + emotion vs LightGBM (Standard)", "M4v6_R1", "M5v6_R1"),
         ("LightGBM + emotion vs LightGBM (Real-time)", "M4v6", "M5v6"),
         ("Neural + emotion vs Neural", "M6_nrms", "M7b_nrms_emotion_gated"),
         ("Combined + emotion vs Combined (Standard, same weight)", "B0_R1", "B1_R1_same_weight"),
         ("Combined + emotion vs Combined (Real-time, same weight)", "B0", "B1_same_weight")]
# same-weight blends: name -> (twin blend, its weight file and key, tree with emotion, NRMS with emotion)
SAME_WEIGHT = {"B1_R1_same_weight": ("B0_R1", "v7_blends_choice.json", "B0_R1_w", "M5v6_R1", "M7b_nrms_emotion_gated", "M4v6_R1", "M6_nrms"),
               "B1_same_weight": ("B0", "v6_blends_choice.json", "B0_w", "M5v6", "M7b_nrms_emotion_gated", "M4v6", "M6_nrms")}


def saved_scores(code):
    return pd.read_parquet(RESULTS / "scores" / f"{code}__dev.parquet")


def blend(truth, w, tree, nrms):
    bl = importlib.import_module("07_04_blends")
    imp = truth["imp_key"].to_numpy()
    z = w * bl.zscore(imp, bl.scores(tree, "dev", truth)) + (1 - w) * bl.zscore(imp, bl.scores(nrms, "dev", truth))
    return truth[["imp_key", "news_id"]].assign(score=z)


def dataset_overview():
    imps = pd.read_parquet(PROCESSED / "impressions.parquet", columns=["imp_key", "user_id", "part", "label", "time"])
    rows = []
    for part, name in (("fit", "Train"), ("valid", "Validate"), ("dev", "Test")):
        t = imps[imps["part"].astype(str) == part]
        day = pd.to_datetime(t["time"]).dt.normalize()
        rows.append({"part": part, "name": name, "first_day": day.min().strftime("%Y-%m-%d"), "last_day": day.max().strftime("%Y-%m-%d"),
                     "impressions": t["imp_key"].nunique(), "users": t["user_id"].nunique(), "candidate_rows": len(t),
                     "clicks": int(t["label"].sum())})
    out = pd.DataFrame(rows)
    out["articles_total"] = len(pd.read_parquet(PROCESSED / "news.parquet", columns=["news_id"]))
    out.to_csv(TABLES / "final_dataset_overview.csv", index=False)
    return out


def main():
    t0 = time.time()
    truth = load_truth("dev")
    news = pd.read_parquet(PROCESSED / "news.parquet", columns=["news_id", "category"]).set_index("news_id")["category"].astype(str)
    emo = pd.read_parquet(PROCESSED / "news_emotion_labels_variants.parquet", columns=["news_id", "label_v2b"]).set_index("news_id")["label_v2b"]
    n_articles = truth["news_id"].nunique()
    per, cover, rows = {}, {}, []
    for code, (name, fam, setting, emotion) in FINAL.items():
        if code in SAME_WEIGHT:
            twin, f, key, tree, nrms, tree0, nrms0 = SAME_WEIGHT[code]
            w = float(json.loads((TABLES / f).read_text(encoding="utf-8"))[key])
            check, _ = evaluate_with_lists(truth, blend(truth, w, tree0, nrms0), news, emo)       # rebuild the twin: must equal the saved one
            saved = pd.read_parquet(RESULTS / "per_impression" / f"{twin}__dev.parquet")
            assert np.allclose(check["auc"], saved["auc"], atol=1e-9, equal_nan=True), f"rebuilt {twin} differs from the saved blend"
            scores, source = blend(truth, w, tree, nrms), f"same-weight blend (w_tree = {w}) of saved dev scores {tree} + {nrms}"
        else:
            scores, source = saved_scores(code), f"saved dev scores results/scores/{code}__dev.parquet"
        p, shown = evaluate_with_lists(truth, scores, news, emo)
        saved_per = RESULTS / "per_impression" / f"{code}__dev.parquet"
        if saved_per.exists():                       # identical to the main evaluation
            s = pd.read_parquet(saved_per)
            for k in RANKING:
                assert np.allclose(p[k], s[k], atol=1e-12, equal_nan=True), f"{code} {k} differs from the saved per-impression file"
        per[code], cover[code] = p, len(shown) / n_articles
        rows.append({"model": code, "display_name": name, "family": fam, "setting": setting, "emotion": emotion,
                     **{k: float(p[k].mean()) for k in PER_IMP}, "coverage@10": cover[code], "impressions": len(p), "source": source})
        print(f"{code:28s} auc {rows[-1]['auc']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(TABLES / "final_all_metrics.csv", index=False)
    g = twin_gains(PAIRS, per, load_users("dev"), cover)
    g.insert(1, "setting", [FINAL[b][2] for b in g["emotion_model"]])
    g.to_csv(TABLES / "final_twin_gains_all_metrics.csv", index=False)
    dataset_overview()
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
