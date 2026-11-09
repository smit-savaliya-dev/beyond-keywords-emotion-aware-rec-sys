# Final results table (all models, plan v7)

MIND-small. **Valid** = 14 Nov (for M4v6 / M5v6 and the R1 models: the fold-F4 prediction, whose early stopping used the same day, so it
is optimistic). **Dev** = 15 Nov, the final test. Dev was already seen in weeks 7 and 8, so every week-8 and v7 dev number is a further look;
the primary evidence for the tree models is the rolling-origin folds F1-F4 (days 11-14). Gain = dev AUC of the model minus its twin
(per-impression paired t-test); Holm p is within the declared family; dz = Cohen's dz of the per-impression differences.
Every model has exactly one dev run in `results/tables/runs_log.csv`. Plans: `docs/experiments_plan_v6.md`, `docs/experiments_plan_v7.md`.

**Regimes.** R2 REAL-TIME: behavioural features may use earlier clicks and impressions of the same day (M4v6, M5v6, B0, B1).
R1 STANDARD (primary, declared before results): previous days only. All other models (M0-M7) use no same-day events, so they are the same in both.

| Model | Regime | Emotion | Valid AUC | Dev AUC | Twin | Gain vs twin (dev) | Holm p | dz | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| M0 popularity | both | no | 0.5740 | 0.5890 | - | - | - | - | baseline |
| M1 TF-IDF | both | no | 0.5900 | 0.5744 | - | - | - | - | baseline |
| M2 Sentence-BERT | both | no | 0.6468 | 0.6253 | - | - | - | - | content baseline |
| M3 = M2 + emotion | both | yes | 0.6499 | 0.6282 | M2 | +0.0029 | <0.001 | 0.034 | small, real gain for a content model |
| M3b+ (late fusion) | both | yes | 0.6503 | 0.6293 | M2 | +0.0040 | <0.001 | 0.047 | small, real gain for a content model |
| M4 LightGBM (week 6) | both | no | 0.6745 | 0.6739 | - | - | - | - | old tree baseline |
| M5 = M4 + emotion | both | yes | 0.6887 | 0.6709 | M4 | -0.0030 | <0.001 | -0.017 | worse: item memorising (audit v6) |
| M5b | both | yes | 0.6838 | 0.6647 | M4 | -0.0092 | <0.001 | -0.046 | worse |
| M4v6 | R2 | no | 0.7251 | **0.7111** | M4 | +0.0371 | <0.001 | 0.136 | big gain, but it needs same-day clicks |
| M5v6 (F2) | R2 | yes | 0.7256 | 0.7118 | M4v6 | +0.0007 | 0.093 | 0.008 | too small, not significant |
| M4v6_R1 | R1 | no | 0.6762 | 0.6645 | M4 | -0.0094 | <0.001 | -0.059 | below the old M4 on dev |
| M5v6_R1 (F3) | R1 | yes | 0.6815 | 0.6696 | M4v6_R1 | **+0.0051** | 3e-39 | 0.049 | beats its twin (Rule 8 passed), still below old M4 |
| M6 NRMS | both | no | 0.6407 | 0.6082 | - | - | - | - | reproduced, weaker than trees |
| M7 (M7b, gated) | both | yes | 0.6410 | 0.6085 | M6 | +0.0002 | 0.33 | 0.004 | no effect |
| M7ctrl (emotion permuted) | both | control | 0.6407 | 0.6091 | M7 | M7 - M7ctrl -0.0006 | 0.14 | -0.007 | no emotion information effect |
| B0 = M4v6 + M6 | R2 | no | 0.7287 | 0.7075 | M4v6 | -0.0036 | <0.001 | -0.034 | blending does not help |
| B1 = M5v6 + M7 | R2 | yes | 0.7291 | **0.7120** | B0 | +0.0045 (same weight: +0.0008) | 6e-34 | 0.045 | best dev AUC; gain mostly a weight difference |
| B0_R1 = M4v6_R1 + M6 | R1 | no | 0.6852 | 0.6628 | M4v6_R1 | -0.0017 | <0.001 | -0.013 | blending does not help |
| B1_R1 = M5v6_R1 + M7 | R1 | yes | 0.6896 | 0.6689 | B0_R1 | +0.0061 (same weight: +0.0023 / +0.0032) | 1e-55 | 0.058 | partly a weight difference |

Holm families: R2 primary (plan v6) = M5v6 - M4v6, M7 - M6, M7 - M7ctrl, B1 - B0. R1 primary (plan v7) = M5v6_R1 - M4v6_R1, B1_R1 - B0_R1,
M7 - M6, M7 - M7ctrl. Older rows use their own week's families. Tables: `v7_dev_families.csv`, `v6_final_dev_families.csv`,
`m2m3_comparison.csv`, `m4m5_comparison.csv`, `behaviour_e7_final_family.csv`.

## Does M4v6's +0.037 survive without same-day information?
**No.** On the folds, M4v6 beats the old M4 columns by +0.068 (mean of 4 folds); with previous-days-only features the gain is +0.0044:
**6% survives**. On the final test, M4v6_R1 is **-0.0094 below** the old M4 (0.6645 vs 0.6739), so **none survives** (-25%).
Why: same-day item CTR alone ranks valid at AUC 0.647; the same CTR from previous days only reaches 0.557, below old-style past popularity
(0.574) (`v7_single_feature_auc.csv`). The R1 twin loses on dev mainly on candidates already shown before the day (-0.013 vs old M4) and is
level on new ones (+0.001) (post-hoc, `v7_posthoc_seen_vs_new.csv`).

## Does emotion beat its twin?
| | R1 standard (primary) | R2 real-time |
|---|---|---|
| Twin / emotion model | M4v6_R1 / M5v6_R1 (F3) | M4v6 / M5v6 (F2) |
| Folds F1-F4 pooled: gain (95% user CI) | +0.0075 (0.0068 to 0.0081) | +0.0010 (0.0005 to 0.0014) |
| Folds: p / Holm p (R1 and R2 together) / dz | 7e-109 / 1e-108 / 0.062 | 2e-5 / 2e-5 / 0.012 |
| Per-fold gain F1, F2, F3, F4 | +0.0129, +0.0052, +0.0049, +0.0050 | +0.0036, +0.0011, -0.0006, +0.0007 |
| Folds won (F2-F4); seed-noise SD of twin | 3 of 3; 0.0021 | 2 of 3; 0.0010 |
| Rule 8 | passed (F2, F3, F6 all eligible; F3 highest) | passed by a hair (F2) |
| Dev gain (95% user CI) | +0.0051 (0.0043 to 0.0060) | +0.0007 (-0.0001 to 0.0015) |
| Dev Holm p / dz | 3e-39 / 0.049 | 0.093 / 0.008 |
| Verdict | **beats its twin**, small effect | **too small** |

Shuffled-label controls: fixed-round retrains on F4 0.499-0.507; dev files scored against shuffled labels 0.501-0.502.

## Where emotion helps (R1, dev, exploratory, all declared segments in `v7_subgroups.csv`)
- Consistency: low +0.0040, mid +0.0039, high +0.0069 (Holm p < 0.001 each): more gain for consistent users, as in week 7.
- Persona: excitement + joy +0.0041, fear + nervousness +0.0073, annoyance + optimism +0.0039: the week-7 joy-reader loss does not return.
- Clicked category: tv +0.019, news +0.011, sports +0.009; music -0.042, finance -0.004 (n.s.), lifestyle +0.001 (n.s.).
- R1 gain is larger than R2 gain in every consistency and persona group (R1 minus R2 +0.002 to +0.008).
- Users with no known history also gain (+0.011): part of the R1 effect is item-level emotion (an item's distance from the average emotion
  profile), not personal matching.

## Limitations
- **Dev is a second (third) look**: weeks 7 and 8 saw dev; folds are the primary evidence; fold early stopping uses each fold's test day.
- **Same-day information in R2**: M4v6 / M5v6 / B0 / B1 use earlier clicks of the scored day (on dev: dev-day labels). This is time-causal
  (online setting) but not the standard offline protocol; their numbers cannot be compared with standard MIND results.
- **R1 twin is weak on dev**: below the old M4, so part of the R1 emotion gain may be repairing a poorly transferring twin; the R1 emotion
  model is also below the old M4 (0.6696 vs 0.6739).
- **Item-level emotion in R1**: the gain for users without history shows the R1 emotion columns carry item-level information.
- **GoEmotions on headlines**: trained on Reddit comments; news headlines and abstracts are a domain shift (SemEval-2007 check: moderate agreement).
- **MIND has no ads**: any claim about ad ranking is a transfer argument only.
- **Item cold start untestable**: 21-29% of candidates were never shown before their day, and MIND gives no later outcome for them; NRMS is
  the only model that scores them from text alone, and it is weak here.
- One test day (15 Nov), one dataset, NRMS with mpnet input embeddings (GloVe unavailable, D80), NRMS trained on fit only.

## Claims allowed
- In the standard offline setting (R1), user-relative emotion features beat a fairly tuned twin on every rolling fold and on the final test,
  by a small amount (+0.005 to +0.008 AUC, dz about 0.05).
- In the real-time setting (R2), where same-day click-through is available, emotion adds nothing significant (+0.0007).
- Emotion helps content models a little (M3, M3b+: +0.003 to +0.004) and does not help NRMS (M7).
- M4v6's large gain comes from same-day click-through, not from emotion or from better long-term features.
- Emotion's gain is somewhat larger for emotionally consistent users (descriptive).

## Claims not allowed
- That emotion makes a recommender better overall: the best R1 model (M5v6_R1, 0.6696) is still below the week-6 M4 (0.6739) on dev.
- That M4v6 / B1 (0.711 / 0.712) are comparable to standard offline MIND results: they use same-day labels.
- That the R1 gain is purely personal: part of it is item-level emotion.
- That B1 - B0 (or B1_R1 - B0_R1) measures emotion: it mixes in different blend weights.
- Anything about ad ranking or about a user's emotional state.
