# Final charts

Built by `python scripts/07_final_figures.py` from saved tables only (no training, no dev scoring). Every PNG has a CSV
next to it with the plotted numbers and the exact label text. Sources are in `results/tables/`. Older charts are in
`archive/figures/` (D99). Names and colours: `src/plot_style.py`. Main results use the standard setting (pre-declared
primary); the real-time setting appears only where the two settings are compared. "Final test" = 15 Nov (dev).
"Slide" = where the chart fits a talk; "Report section" = where it fits the written report (D102-D104).

| Chart | Slide | Report section | Takeaway | Source |
|---|---|---|---|---|
| `1_method/pipeline.png` | How it works | Methodology: system design | Articles get emotion scores, readers get an emotion profile, and their match is added to the content score. | method; src/config.py (history length) |
| `1_method/time_split.png` | How it works | Methodology: evaluation design | Train on 126,695 impressions, tune on 30,270, test once on 73,152. | final_dataset_overview.csv |
| `1_method/model_pairs.png` | Models | Methodology: models | Each emotion model is compared with a twin that is identical except for emotion. | names only (src/plot_style.py) |
| `2_data/dataset_overview.png` | Problem | Data: dataset description | 65,238 articles; the test day alone has 73,152 impressions. | final_dataset_overview.csv |
| `2_data/neutral_fix.png` | Problem | Methodology: emotion labelling | The old rule called 81% of articles neutral; the fixed rule calls 39% neutral, so joy, sadness and surprise become visible. | story_emotion_mix.csv (checked against emotion_weighted_vs_combined.json, emotion_label_variants_grid.csv, emotion_v2_choice.json) |
| `2_data/emotion_validation.png` | How it works | Methodology: emotion labelling (validity) | Our scores agree with human ratings (Spearman 0.19 to 0.72, highest for valence), higher than the NRC word lists on every emotion. | semeval2007_validation.csv |
| `2_data/emotion_match_signal.png` | Problem | Data analysis: is there an emotion signal? | Before any model: clicked articles match the reader's emotion profile better than skipped ones (personal match dz 0.25 training, 0.24 validation). | emotion_signal_v2.csv |
| `3_results/all_models_auc.png` | Result 1 | Results: main model comparison | All final models beat random; the best in the standard setting is LightGBM + emotion at 0.670. | final_all_metrics.csv, summary.csv |
| `3_results/all_models_auc_every_run.png` | Backup | Results or appendix: every model run | All 21 runs scored on the test day, including earlier versions and controls, each labelled with its setting and code. | final_all_metrics.csv, summary.csv |
| `3_results/all_metrics_heatmap.png` | Result 1 | Results: other ranking metrics | MRR, nDCG, precision, recall and hit rate order the model families the same way as AUC. | final_all_metrics.csv |
| `3_results/emotion_gain_all_metrics.png` | Result 1 | Results: other ranking metrics | Standard setting: every gain is positive; significant on all eight metrics for LightGBM + emotion, mostly not significant beyond AUC for the meaning and neural models. | final_twin_gains_all_metrics.csv |
| `3_results/emotion_gain_auc.png` | Result 1 | Results: main finding (emotion vs twin) | Emotion adds +0.0029 (meaning + emotion), +0.0040 (meaning + emotion, tuned profile), +0.0051 (lightgbm + emotion, standard setting), +0.0007 (lightgbm + emotion, real-time setting), +0.0002 (neural + emotion). | final_twin_gains_all_metrics.csv |
| `3_results/two_settings.png` | Same-day check | Results: the two settings | Emotion adds +0.0051 in the standard setting and +0.0007 (not significant) in real time. | final_all_metrics.csv, final_twin_gains_all_metrics.csv |
| `3_results/two_settings_all_metrics.png` | Same-day check | Results / discussion: the two settings | Standard: every metric gains significantly. Real-time: AUC is flat and emotion significantly lowers nDCG@5, nDCG@10, Precision@5, Recall@10, Hit rate@5. | final_twin_gains_all_metrics.csv |
| `3_results/same_day_check.png` | Same-day check | Discussion: why real-time scores higher | LightGBM scores 0.711 with same-day clicks and 0.665 without; the click rate alone falls from 0.647 to 0.557. | final_all_metrics.csv, v7_single_feature_auc.csv, summary.csv |
| `3_results/fold_evidence.png` | Result 1 | Results: robustness | In the standard setting the gain is positive on all four folds; in real time it stays near the noise. | v7_fold_evidence.csv, v7_fold_means.csv |
| `3_results/controls.png` | Backup | Results: validity checks | With clicks shuffled every ranker scores 0.501-0.502 (chance); shuffled emotion scores like real emotion in the neural ranker. | final_all_metrics.csv, v7_dev_shuffled_controls.csv, summary.csv |
| `4_analysis/gain_by_consistency.png` | Result 2 | Discussion: who benefits | Positive for all readers, largest for the most consistent (+0.0069 vs +0.0040). | v7_subgroups.csv |
| `4_analysis/gain_by_category.png` | Result 2 | Discussion: where emotion helps | Largest when the click is tv (+0.0187); music clicks lose (-0.0420). | v7_subgroups.csv |
| `4_analysis/not_topic.png` | Result 2 | Discussion: emotion is not just topic | Inside one category the emotion match still beats chance (AUC 0.555 vs 0.500): it is not just topic. | behaviour_e5_within_category_auc.csv, behaviour_e5_m3_gain_within_category.csv |
| `4_analysis/list_quality.png` | Result 2 | Discussion: list quality (diversity, coverage) | Emotion changes the variety of the top 10 by about 1% or less; category variety falls slightly but significantly for LightGBM, Neural, Combined. | final_all_metrics.csv, final_twin_gains_all_metrics.csv |
| `overview.png` | Conclusion | Summary / poster | One-page summary of the six main charts. | the six chart PNGs |

## EDA (kept as it is)

`EDA/` holds the exploratory charts, written by other scripts and not by this one (D102):

| Chart | Report section | Written by |
|---|---|---|
| `EDA/01_part_sizes.png` ... `EDA/06_integrity.png`, `EDA/eda_overview.png`, `EDA/eda_stats.json` | Data: exploratory analysis | `scripts/01_04_eda.py` |
| `EDA/emotion_cross_source_spearman.png` | Methodology: emotion sources agree | `scripts/02_05_diagnostics.py` |
| `EDA/m5_feature_importance.png` | Appendix only: the EARLIER week-4 LightGBM + emotion (M5), not a final model | `scripts/05_03_feature_importance.py` |

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
