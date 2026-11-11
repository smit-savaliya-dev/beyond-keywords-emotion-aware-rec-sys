# Final charts

Built by `python scripts/07_final_figures.py` from saved tables only (no training, no dev scoring). Every PNG has a CSV
next to it with the plotted numbers and the exact label text. Sources are in `results/tables/`. Older charts are in
`archive/figures/` (D99). Names and colours: `src/plot_style.py`. Main results use the standard setting (pre-declared
primary); the real-time setting appears only where the two settings are compared. "Final test" = 15 Nov (dev).

| Chart | Slide | Takeaway | Source |
|---|---|---|---|
| `1_method/pipeline.png` | How it works | Articles get emotion scores, readers get an emotion profile, and their match is added to the content score. | method; src/config.py (history length) |
| `1_method/time_split.png` | How it works | Train on 126,695 impressions, tune on 30,270, test once on 73,152. | final_dataset_overview.csv |
| `1_method/model_pairs.png` | Models | Each emotion model is compared with a twin that is identical except for emotion. | names only (src/plot_style.py) |
| `2_data/dataset_overview.png` | Problem | 65,238 articles; the test day alone has 73,152 impressions. | final_dataset_overview.csv |
| `2_data/neutral_fix.png` | Problem | The old rule called 81% of articles neutral; the fixed rule calls 39% neutral. | emotion_weighted_vs_combined.json, emotion_label_variants_grid.csv, emotion_v2_choice.json |
| `2_data/emotion_validation.png` | How it works | Our scores agree with human ratings (Spearman 0.19 to 0.59), better than the NRC word list. | semeval2007_validation.csv |
| `3_results/all_models_auc.png` | Result 1 | All final models beat random; the best in the standard setting is LightGBM + emotion at 0.670. | final_all_metrics.csv, summary.csv |
| `3_results/all_metrics_heatmap.png` | Result 1 | MRR, nDCG, precision, recall and hit rate rank the models almost the same way as AUC. | final_all_metrics.csv |
| `3_results/emotion_gain_all_metrics.png` | Result 1 | Where emotion helps on AUC it helps on the other metrics too; no metric shows a significant loss. | final_twin_gains_all_metrics.csv |
| `3_results/emotion_gain_auc.png` | Result 1 | Emotion adds +0.0029 (meaning + emotion), +0.0040 (meaning + emotion, tuned profile), +0.0051 (lightgbm + emotion, standard setting), +0.0007 (lightgbm + emotion, real-time setting), +0.0002 (neural + emotion). | final_twin_gains_all_metrics.csv |
| `3_results/two_settings.png` | Same-day check | Emotion adds +0.0051 in the standard setting and +0.0007 (not significant) in real time. | final_all_metrics.csv, final_twin_gains_all_metrics.csv |
| `3_results/same_day_check.png` | Same-day check | LightGBM scores 0.711 with same-day clicks and 0.665 without; the click rate alone falls from 0.647 to 0.557. | final_all_metrics.csv, v7_single_feature_auc.csv, summary.csv |
| `3_results/fold_evidence.png` | Result 1 | In the standard setting the gain is positive on all four folds; in real time it stays near the noise. | v7_fold_evidence.csv, v7_fold_means.csv |
| `4_analysis/gain_by_consistency.png` | Result 2 | Positive for all readers, largest for the most consistent (+0.0069 vs +0.0040). | v7_subgroups.csv |
| `4_analysis/gain_by_category.png` | Result 2 | Largest when the click is tv (+0.0187); music clicks lose (-0.0420). | v7_subgroups.csv |
| `4_analysis/not_topic.png` | Result 2 | Inside one category the emotion match still beats chance (AUC 0.555 vs 0.500): it is not just topic. | behaviour_e5_within_category_auc.csv, behaviour_e5_m3_gain_within_category.csv |
| `4_analysis/list_quality.png` | Result 2 | Emotion leaves the variety of categories and emotions in the top 10 almost unchanged. | final_all_metrics.csv, final_twin_gains_all_metrics.csv |
| `overview.png` | Conclusion | One-page summary of the six main charts. | the six chart PNGs |

## Names used in charts

| Display name | Code |
|---|---|
| Popularity | M0 |
| Keyword match (TF-IDF) | M1 |
| Meaning match (Sentence-BERT) | M2 |
| Meaning + emotion | M3 |
| Meaning + emotion, tuned profile | M3b+ |
| LightGBM ranker | M4v6 |
| LightGBM + emotion | M5v6 |
| Neural ranker (NRMS) | M6 |
| Neural + emotion | M7 |
| Combined ranker (LightGBM + neural) | B0 |
| Combined + emotion | B1 |
| Standard setting (previous days only) | R1 |
| Real-time setting (with same-day clicks) | R2 |

## Metrics in simple words

- **AUC**: the chance that a clicked article is ranked above a non-clicked one in the same list; 0.50 = random, 1 = perfect.
- **MRR**: 1 divided by the position of the clicked article (1 = first place), averaged over clicks.
- **nDCG@5 / nDCG@10**: how good the order of the top 5 / top 10 is, giving more credit to clicks near the top (0 to 1).
- **Precision@5**: share of the top 5 that the user clicked.
- **Recall@10**: share of the user's clicks that appear in the top 10.
- **Hit rate@5 / @10**: share of lists with at least one click in the top 5 / top 10.
- **Category / emotion diversity@10**: number of different news categories / emotion labels in the top 10.
- **Catalogue coverage@10**: share of all test-day candidate articles that reach at least one top 10.
- **Gain**: emotion model minus its twin without emotion. **Holm p**: p-value corrected for testing many gains at once;
  above 0.05 = not significant. **CI**: 95% confidence interval.
