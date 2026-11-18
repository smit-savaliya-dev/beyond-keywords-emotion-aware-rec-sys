# Week-9 figures

Copies of every figure made or updated by `python scripts/08_week9_figures.py` (originals in `3_results/`, `1_method/` and the
figures root). Built only from saved tables (`scripts/08_04_week9_metrics.py`), test day = 15 Nov, standard setting unless
stated. Models: one per approach, chosen on validation AUC (D109). PNGs only here; the CSV with the plotted numbers sits next
to each original.

| Figure | Takeaway | Source CSV |
|---|---|---|
| `00_week9_overview.png` | One 16:9 page with figures 01, 02, 03, 06, 10 and 11. | the six chart PNGs |
| `01_metrics_table.png` | LightGBM + emotion is best in 6 of 8 columns; every emotion model is above its twin in every column. | final_metrics_all_models.csv |
| `02_emotion_gain_table.png` | 29 of 32 gains are significant and positive, none is a significant loss; the not-significant ones are Meaning match P@5, Meaning match HR@5, Meaning match HR@10. | final_emotion_gains_all_metrics.csv |
| `03_auc.png` | best is LightGBM + emotion (0.670); emotion helps significantly for Meaning match, LightGBM, Neural (NRMS), Combined. | final_metrics_all_models.csv, final_emotion_gains_all_metrics.csv |
| `04_mrr.png` | best is LightGBM + emotion (0.325); emotion helps significantly for Meaning match, LightGBM, Neural (NRMS), Combined. | final_metrics_all_models.csv, final_emotion_gains_all_metrics.csv |
| `05_ndcg5.png` | best is LightGBM + emotion (0.358); emotion helps significantly for Meaning match, LightGBM, Neural (NRMS), Combined. | final_metrics_all_models.csv, final_emotion_gains_all_metrics.csv |
| `06_ndcg10.png` | best is LightGBM + emotion (0.420); emotion helps significantly for Meaning match, LightGBM, Neural (NRMS), Combined. | final_metrics_all_models.csv, final_emotion_gains_all_metrics.csv |
| `07_precision5.png` | best is LightGBM + emotion (0.129); emotion helps significantly for LightGBM, Neural (NRMS), Combined; Meaning match: not significant. | final_metrics_all_models.csv, final_emotion_gains_all_metrics.csv |
| `08_recall10.png` | best is Combined + emotion (0.677); emotion helps significantly for Meaning match, LightGBM, Neural (NRMS), Combined. | final_metrics_all_models.csv, final_emotion_gains_all_metrics.csv |
| `09_hitrate.png` | Hit rate@5: best is LightGBM + emotion (0.581); emotion helps significantly for LightGBM, Neural (NRMS), Combined; Meaning match: not significant. Hit rate@10: best is Combined + emotion (0.757); emotion helps significantly for LightGBM, Neural (NRMS), Combined; Meaning match: not significant. | final_metrics_all_models.csv, final_emotion_gains_all_metrics.csv |
| `10_two_settings.png` | Emotion adds +0.0051 AUC in the standard setting (significant) and +0.0007 with same-day clicks (not significant). | final_metrics_all_models.csv, final_emotion_gains_all_metrics.csv |
| `11_emotion_gain_auc.png` | Emotion adds +0.0040 (Meaning match), +0.0051 (LightGBM), +0.0053 (Neural (NRMS)), +0.0023 (Combined) AUC; all significant; the 95% intervals overlap, so no approach clearly gains more than another. | final_emotion_gains_all_metrics.csv |
| `12_metrics_explained.png` | A worked example (10 articles, clicks at positions 2 and 7) showing what AUC, MRR, nDCG@10, P@5, R@10 and HR@10 measure. | computed with src/metrics.py on the example list |

## Names used

| Display name | Internal code |
|---|---|
| Popularity | M0_popularity |
| Keyword match (TF-IDF) | M1_tfidf |
| Meaning match (Sentence-BERT) | M2_sbert |
| Meaning + emotion | M3b_plus (tuned profile) |
| LightGBM ranker | M4v6_R1 (standard); M4v6 (real-time, figure 10) |
| LightGBM + emotion | M5v6_R1 (standard); M5v6 (real-time, figure 10) |
| Neural ranker (NRMS) | M6_nrms |
| Neural + emotion | M7d_nrms_emotion_fusion |
| Combined ranker | B0_R1 |
| Combined + emotion | B1_R1_same_weight (B1_R1 at B0_R1's blend weight) |
| previous days only (standard setting) | R1 |
| with same-day clicks (real-time setting) | R2 |

Every earlier version and its validation and test-day scores: `results/tables/all_versions_full.csv` (not in figures).

## Metric glossary

- **AUC**: chance that a clicked article is ranked above a non-clicked one in the same list; 0.50 = random, 1 = perfect.
- **MRR**: 1 / position of the clicked article (1 = first place), averaged over clicks: how high the clicks appear.
- **nDCG@5 / nDCG@10**: quality of the order of the top 5 / top 10, with more credit for clicks near the top (0 to 1).
- **P@5 (Precision@5)**: share of the top 5 that were clicked.
- **R@10 (Recall@10)**: share of the user's clicks found in the top 10.
- **HR@5 / HR@10 (Hit rate)**: share of lists with at least one click in the top 5 / top 10.
- **Gain**: emotion model minus its twin without emotion, on the same test-day lists. **95% CI**: clustered by user.
  **Holm p**: paired t-test p corrected for the 32 tests (4 approaches x 8 metrics); above 0.05 = not significant (n.s.).
