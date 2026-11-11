"""Stage 0 audit (week 8): item overlap, seed noise floor of M4 and M5, M4 feature inventory and ablation, tuning parity, valid-day vs
dev-day differences, and how much M5 relied on raw item-level emotion. Fit and valid only for every model fit; dev is read as metadata
(candidate counts, history lengths, categories, item ids), never as model output.

Reads:  impressions, behaviours, news, the M4 feature tables (fit, valid), frozen M4 and M5 settings, m5_feature_importance.csv, runs_log.csv.
Writes: results/tables/audit_v6_*.csv; archive/figures/v6_improve/audit_*.png.
Run:    python scripts/07_00_audit.py
"""
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import behaviour_charts as bc
from src.config import ARCHIVE_FIGS, FIGURES, FROZEN_DIR, LTR_FEATURE_FILE, LTR_GRID, M4_FEATURES, M4_NAME, M5_NAME, PROCESSED, TABLES
from src.evaluate import load_truth
from src.lightgbm_ranker import file_sha, fit_or_load, model_features, score_frame

OUT = ARCHIVE_FIGS / "v6_improve"
SEEDS = [42, 43, 44, 45, 46]
GROUPS = {"popularity (pop_ctr)": ["pop_ctr"], "TF-IDF match (m1_score)": ["m1_score"], "Sentence-BERT match (m2_score)": ["m2_score"],
          "category and subcategory": ["category", "subcategory"], "user category share": ["user_cat_affinity"],
          "context (history, title, list size)": ["history_len", "title_len", "imp_size"]}
ITEM_RAW = ("go_pct_", "go_vad_", "go_dist_", "vad_", "nrc_")


def overlap():
    rows = pd.read_parquet(PROCESSED / "impressions.parquet", columns=["part", "news_id", "label"])
    part = rows["part"].astype(str)
    seen_fit = set(rows.loc[part == "fit", "news_id"].astype(str))
    seen_fv = seen_fit | set(rows.loc[part == "valid", "news_id"].astype(str))
    out = []
    for p, seen, ref in (("valid", seen_fit, "fit"), ("dev", seen_fv, "fit + valid"), ("dev", seen_fit, "fit")):
        ids = rows.loc[part == p, "news_id"].astype(str)
        clicked = rows.loc[(part == p) & (rows["label"] == 1), "news_id"].astype(str)
        out.append({"part": p, "seen_in": ref, "candidate_rows_seen": float(ids.isin(seen).mean()), "unique_items_seen": float(pd.Series(ids.unique()).isin(seen).mean()),
                    "clicked_rows_seen": float(clicked.isin(seen).mean())})
    return pd.DataFrame(out)


def day_profile():
    beh = pd.read_parquet(PROCESSED / "behaviors.parquet", columns=["imp_key", "part", "history_len", "time"])
    rows = pd.read_parquet(PROCESSED / "impressions.parquet", columns=["imp_key", "part", "news_id", "label"])
    news = pd.read_parquet(PROCESSED / "news.parquet", columns=["news_id", "category"]).set_index("news_id")["category"]
    out, cats = [], []
    beh["day"] = pd.to_datetime(beh["time"]).dt.strftime("%d %b")
    for p in ("fit", "valid", "dev"):
        b, r = beh[beh["part"].astype(str) == p], rows[rows["part"].astype(str) == p]
        size = r.groupby("imp_key").size()
        out.append({"part": p, "impressions": len(b), "mean_candidates": float(size.mean()), "median_candidates": float(size.median()),
                    "mean_clicks": float(r["label"].sum() / len(b)), "mean_history": float(b["history_len"].mean()),
                    "cold_start_share": float((b["history_len"] == 0).mean())})
        c = r.loc[r["label"] == 1, "news_id"].astype(str).map(news).value_counts(normalize=True).rename("click_share")
        cats.append(c.to_frame().assign(part=p).reset_index(names="category"))
    return pd.DataFrame(out), pd.concat(cats, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ov = overlap()
    ov.to_csv(TABLES / "audit_v6_overlap.csv", index=False)
    days, cats = day_profile()
    days.to_csv(TABLES / "audit_v6_day_profile.csv", index=False)
    cats.to_csv(TABLES / "audit_v6_click_categories.csv", index=False)
    print(ov.round(4).to_string(index=False), days.round(3).to_string(index=False), sep="\n")

    fit_path, valid_path = (PROCESSED / LTR_FEATURE_FILE.format(part=p) for p in ("fit", "valid"))
    fit, valid = pd.read_parquet(fit_path), pd.read_parquet(valid_path)
    sha = file_sha(fit_path, valid_path)
    truth = load_truth("valid", PROCESSED)
    rows = []
    for model in (M4_NAME, M5_NAME):
        cfg = json.loads((FROZEN_DIR / f"{model}.json").read_text(encoding="utf-8"))["config"]
        params = {k: cfg[k] for k in LTR_GRID}
        for seed in SEEDS:
            booster, info = fit_or_load(fit, valid, model_features(model), params, seed, "", sha, resume=True)
            _, _, s = score_frame(booster, valid, model_features(model), truth)
            rows.append({"model": model, "seed": seed, "auc": float(s["auc"]), "rounds": info["best_iteration"]})
            print(rows[-1], flush=True)
    noise = pd.DataFrame(rows)
    noise.to_csv(TABLES / "audit_v6_seed_noise.csv", index=False)
    summ = noise.groupby("model")["auc"].agg(["mean", "std", "min", "max"])
    print(summ)

    m4 = json.loads((FROZEN_DIR / f"{M4_NAME}.json").read_text(encoding="utf-8"))["config"]
    params = {k: m4[k] for k in LTR_GRID}
    abl = [{"dropped": "none (M4)", "auc": float(noise[(noise["model"] == M4_NAME) & (noise["seed"] == 42)]["auc"].iloc[0])}]
    for name, cols in GROUPS.items():
        feats = [f for f in M4_FEATURES if f not in cols]
        booster, info = fit_or_load(fit, valid, feats, params, 42, "", sha, resume=True)
        _, _, s = score_frame(booster, valid, feats, truth)
        abl.append({"dropped": name, "auc": float(s["auc"])})
        print(abl[-1], flush=True)
    abl = pd.DataFrame(abl)
    abl["change"] = abl["auc"] - abl["auc"].iloc[0]
    abl.to_csv(TABLES / "audit_v6_m4_ablation.csv", index=False)

    imp = pd.read_csv(TABLES / "m5_feature_importance.csv")
    imp["kind"] = np.where(imp["family"] != "emotion", "not emotion",
                           np.where(imp["feature"].str.startswith(ITEM_RAW), "raw item emotion", "user-relative emotion"))
    split = imp.groupby("kind")[["contrib_share", "gain_share"]].sum().reset_index()
    split["n_features"] = imp.groupby("kind").size().to_numpy()
    split.to_csv(TABLES / "audit_v6_m5_importance_split.csv", index=False)
    runs = pd.read_csv(TABLES / "runs_log.csv")
    parity = runs[runs["part"] == "valid"].groupby("model").size().rename("valid_runs_logged").reset_index()
    parity.to_csv(TABLES / "audit_v6_tuning_parity.csv", index=False)
    print(split, parity.to_string(index=False), sep="\n")

    # charts
    o = ov.iloc[:2]
    bc.bars(OUT / "audit_overlap.png", "Valid shares far more items with training than the final test",
            f"Share of candidate rows whose article was already shown in the training days: valid {100 * o['candidate_rows_seen'].iloc[0]:.1f}% "
            f"(seen in fit) vs final test {100 * o['candidate_rows_seen'].iloc[1]:.1f}% (seen in fit or valid). Item-level features can act as item IDs on valid.",
            ["valid (14 Nov)\nvs fit", "final test (15 Nov)\nvs fit + valid"], (100 * o["candidate_rows_seen"]).tolist(), "Candidate rows already seen (%)",
            fmt="{:.1f}%", colors=[bc.BLUE, bc.ORANGE], zero=False, lim=(0, 100))
    bc.bars(OUT / "audit_seed_noise.png", "How much does the score move with the random seed alone?",
            f"Valid AUC of M4 and M5 over 5 seeds (bars = min to max). SD: M4 {summ.loc[M4_NAME, 'std']:.4f}, M5 {summ.loc[M5_NAME, 'std']:.4f}. "
            "A gain smaller than about one SD cannot be told apart from seed noise.",
            ["M4 (no emotion)", "M5 (with emotion)"], summ["mean"].tolist(), "AUC (valid, mean of 5 seeds)", low=summ["min"].tolist(),
            high=summ["max"].tolist(), fmt="{:.4f}", colors=[bc.BLUE, bc.ORANGE], zero=False, base=0.66, lim=(0.66, float(summ["max"].max()) + 0.006))
    a = abl.iloc[1:].sort_values("change")
    bc.bars(OUT / "audit_m4_ablation.png", "What carries M4? Dropping one feature group at a time",
            f"Change in valid AUC when one group is removed from M4 (seed 42; M4 = {abl['auc'].iloc[0]:.4f}). The biggest drops are what the model needs most.",
            a["dropped"].tolist(), a["change"].tolist(), "Change in AUC when dropped (valid)", colors=[bc.BLUE] * len(a), horizontal=True)
    bc.bars(OUT / "audit_m5_importance_split.png", "What M5 actually used",
            "Share of M5's mean absolute contribution by feature kind (valid). Raw item-level emotion columns can act as item IDs, so week 8 drops them.",
            split["kind"].tolist(), (100 * split["contrib_share"]).tolist(), "Share of contribution (%)", fmt="{:.1f}%",
            colors=[bc.GREY if k == "not emotion" else (bc.ORANGE if k == "raw item emotion" else bc.SKY) for k in split["kind"]], zero=False)


if __name__ == "__main__":
    main()
