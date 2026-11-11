# Presentation charts

Built by `python scripts/06_present_charts.py` from saved result tables only (no training, no dev scoring). Each PNG has a CSV
next to it with the plotted values and the exact label text. Source CSVs are in `results/tables/` unless a path is given.
R1 = previous days only (standard setting); R2 = with same-day clicks (real-time setting). "Final test" = 15 Nov (dev).

| Chart | Slide | Takeaway | Source |
|---|---|---|---|
| `01_pipeline.png` | How it works | Articles get emotion scores, readers get an emotion profile, and their match is added to the content score. | method (src/config.py for the history length) |
| `02_time_split.png` | How it works | Train on 126,695 impressions (9-13 Nov), tune on 30,270 (14 Nov), test once on 73,152 (15 Nov). | results/figures/EDA/eda_stats.json |
| `03_neutral_fix.png` | Problem | The old rule called 81% of articles neutral; the fixed rule calls 39% neutral. | emotion_weighted_vs_combined.json, emotion_label_variants_grid.csv, emotion_v2_choice.json |
| `04_model_ladder.png` | Models | Each emotion model is compared with a twin that is identical except for emotion. | model_ladder.csv |
| `05_all_models_auc.png` | Result 1 | All models beat random (0.50); the best standard model is M4 LightGBM at 0.674. | model_ladder.csv |
| `06_emotion_gain_vs_twin.png` | Result 1 | Emotion beats its twin by small amounts; the clearest gain is +0.0051 with previous-days features; the same-day and NRMS gains are not significant. | m2m3_comparison.csv, behaviour_e7_final_family.csv, v7_dev_families.csv |
| `07_gain_by_consistency.png` | Result 2 | The gain is positive for all readers and largest for the most consistent ones (+0.0069 vs +0.0040). | v7_subgroups.csv |
| `08_gain_by_category.png` | Result 2 | Biggest gain when the click is tv (+0.0187); music clicks lose (-0.0420). | v7_subgroups.csv |
| `09_not_topic.png` | Result 2 | Inside a single category the emotion match still beats chance (AUC 0.555 vs control 0.500), so emotion is not only topic. | behaviour_e5_within_category_auc.csv, behaviour_e5_m3_gain_within_category.csv |
| `10_same_day_check.png` | Same-day check | With previous days only, M4v6 drops from 0.711 to 0.665, below the first M4 (0.674); the click rate alone falls from 0.647 to 0.557. | v7_survival_dev.csv, v7_single_feature_auc.csv, model_ladder.csv |
| `11_two_settings.png` | Same-day check | Emotion adds +0.0051 with previous days only and +0.0007 (not significant) with same-day clicks. | v7_dev_families.csv |
| `12_fold_evidence.png` | Conclusion | With previous-days features the gain is above zero on all four folds (+0.0129, +0.0052, +0.0049, +0.0050); with same-day clicks it stays near the noise. | v7_fold_evidence.csv, v7_fold_means.csv |
| `00_overview.png` | Conclusion | One-page summary of charts 05, 06, 07, 08, 10 and 11. | the six chart PNGs |

## Numbers to quote

- M3 vs M2 (content model (Sentence-BERT)): +0.0029 AUC, 95% CI +0.0023 to +0.0035, Holm p < 0.001
- M3b+ vs M2 (content model, week-7 profile): +0.0040 AUC, 95% CI +0.0034 to +0.0046, Holm p < 0.001
- M5v6 vs M4v6 (feature ranker, previous days only (R1)): +0.0051 AUC, 95% CI +0.0043 to +0.0058, Holm p < 0.001
- M5v6 vs M4v6 (feature ranker, with same-day clicks (R2)): +0.0007 AUC, 95% CI +0.0001 to +0.0014, Holm p = 0.09
- M7 vs M6 (neural reader model (NRMS)): +0.0002 AUC, 95% CI -0.0002 to +0.0007, Holm p = 0.33
- Gain by consistency (R1, test day): most varied readers +0.0040, middle +0.0039, most consistent readers +0.0069
- Gain by clicked category (R1, test day): tv +0.0187, news +0.0111, sports +0.0091, other +0.0033, lifestyle +0.0010, finance -0.0035, music -0.0420
- Feature ranker without emotion, test AUC: first M4 0.674, M4v6 with same-day clicks 0.711, M4v6 previous days only 0.665
- 24-hour click rate alone (validation day): with same-day clicks 0.647, previous days only 0.557
- Articles labelled neutral: 81% before the fix, 39% after
- Impressions: train 126,695, validate 30,270, test 73,152
