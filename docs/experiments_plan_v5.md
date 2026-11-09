# Experiments plan v5: when does emotional context help? (pre-declared)

Written and committed before any of these results exist. Anything added after the first result is marked **post-hoc exploratory**.
Branch `week7-behaviour`. Decisions D73 to D76 in `docs/decisions.md`. New feature definitions: `config/emotion_spec_v5.json`
(v4 plus a `behaviour` section; every model input of v4 is unchanged, so M3 and M5 do not change).

## Rules that apply to every experiment
- **Parts.** fit (9-13 Nov) defines cut-offs and explores, valid (14 Nov) chooses, dev (15 Nov) reports. Dev is never used for a choice.
  Dev numbers for subgroups are read from the one saved dev prediction file of a model (`results/scores/`, `results/per_impression/`);
  they are not extra dev runs. New final models (M3b+, M5b) run on dev exactly once (`--final-dev`, the existing guard).
- **Time-causal.** A feature of an impression uses only (a) the history column, which ends before the log period, or (b) clicks and
  impressions with a timestamp strictly earlier than the impression. Each new feature has a pytest leakage test.
- **Emotion vector.** The spec v4 vector: the 8 `go_pct_*` percentiles, title share 0.5 (`news_emotion_features_v2.parquet`).
  Personal match = cos(item, profile) - cos(item, corpus mean), the M3 score (D58).
- **Statistics.** Per-impression paired t-test (primary), mean difference with 95% CI, Cohen's dz, Holm across the experiment's family
  (listed below, per part), a user-level check (per-user mean, paired t-test), and a shuffled-label control: click labels permuted within
  each impression (seed 42), statistic recomputed, which must be near chance (AUC near 0.5, gain and trend near 0).
- **Subgroups.** Gains per group use the subgroup helper (`src/subgroups.py`): mean AUC, MRR, nDCG@10 per group, gain of A over B with a
  95% bootstrap CI that resamples users (1,000 draws, seed 42), and the group size. Groups under 500 impressions are reported and flagged
  "too small", never hidden.
- **Charts.** `results/figures/v5_behaviour/<exp>_<name>.png`, 200 dpi, the style of `scripts/05_presentation_charts.py`, a plain
  title and a one-line caption. Every chart's numbers are saved in `results/tables/behaviour_<exp>*.csv`.
- **Reporting.** Every variant tried is reported, negative results included.

## E1 Emotionally consistent vs varied users (weight 25%)
- **Consistency** (per behaviour row, history only): mean pairwise cosine of the emotion vectors of the last 50 known history items
  (primary). Secondary: mean cosine of each item to the centroid (= 1 - dispersion). Defined for 2 or more known items; others form the
  group "no history".
- **Tertiles**: cut-offs = 1/3 and 2/3 quantiles over fit impressions with 2+ items; fixed for valid and dev.
- **Questions**: gain of M3 over M2 per tertile (valid, then dev from the saved files). Trend: Spearman rho between consistency and the
  per-impression AUC gain, and the high-minus-low tertile difference (Welch t-test). Family E1 per part = {rho, high - low}.
- **M3b (adaptive)**: M3 with alpha chosen per consistency tertile on valid from {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}; "no history" keeps 0.8.
  Valid only (no dev run; its dev test is M3b+). Family E1b on valid = {M3b - M3, M3b - M2}. Valid is optimistic here: it chose alpha.
- **Control**: shuffled labels; and consistency permuted across users (trend must vanish).
- **Charts**: gain by tertile with CIs; AUC per tertile for M2 and M3; consistency histogram.

## E2 Short-window and same-day profiles (weight 20%)
- **Windows** for the emotion profile: last 5, 10, 20, 50 known history items, decay 0.9 or none (8 profiles). For each, M3 fusion with
  the frozen M2 base, personal match from that profile, alpha 0.8. Best by valid AUC; ties (within 0.0001) go to the current profile
  (50, decay 0.9). Also reported, not used: M2 AUC with the same 8 content profiles.
- **Same-day profile**: emotion vectors of the user's clicks in strictly earlier impressions of the same calendar day, newest weight 1,
  decay 0.9. Blend profile = a * long-term + (1 - a) * same-day for impressions with any earlier same-day click ("covered");
  others keep the long-term profile. a in {1.0, 0.8, 0.6, 0.4, 0.2, 0.0}, chosen on valid by AUC.
  **Value rule**: the blend is kept only if the chosen a < 1 beats a = 1 on covered valid impressions (paired t-test, p < 0.05);
  otherwise a = 1. Coverage is reported per part; results are reported for covered and uncovered impressions separately.
- **Family E2 (valid)** = {best window - current, best blend - a = 1 on covered}. Control: shuffled labels.
- **Charts**: AUC vs window length (M2 AUC and M3 gain), coverage bar, gain vs a, covered vs uncovered gain.

## E3 Emotion drift (weight 10%)
- **Drift** (users with 10+ known items): mean pairwise cosine within the last 5 items minus cosine(mean of last 5, mean of the older
  items, up to 45). High = the recent items agree with each other more than with the older ones. Tertiles on fit.
- **Question** (valid): gain of the short window (last 5, no decay) over the long window (last 50, decay 0.9), both as M3 fusion at
  alpha 0.8, per drift tertile. Family E3 = {Spearman rho of drift vs that gain, high - low}. Controls: shuffled labels, drift permuted.
- **Dev**: M3b+ - M3 per drift tertile from the final files.
- **Charts**: drift distribution; short-vs-long gain by drift tertile.

## E4 New articles: item cold start (weight 15%)
- **Exposure** of a candidate row = number of strictly earlier impressions (any part, by timestamp) in which the item was shown.
  Buckets: new (0), few (1 to Q), many (> Q), Q = median exposure of fit rows with exposure > 0.
- **Metric**: per-impression AUC restricted to the candidates of one bucket (impressions with a click and a non-click in that bucket).
- **Models**: M0 to M5 from the saved valid and dev score files. Gains M3 - M2 and M5 - M4 per bucket.
  Family E4 per part = {M3 - M2 gain: new vs many, M5 - M4 gain: new vs many} (Welch t-tests). Control: buckets permuted within impressions.
- **Charts**: AUC by bucket per model; M3 - M2 gain by bucket.

## E5 Emotion within the same topic (weight 10%)
- **(a)** AUC of the personal match alone among candidates of one category inside an impression (fit and valid); mean per category
  (top 10 by fit clicks, rest "other") and overall, tested against 0.5. Dev: M3 - M2 within-category gain from the saved files.
- **(b)** Residual emotion: item vector minus its category mean (means over items shown in fit impressions). Residual match = cosine of the
  residual item vector with the profile of residual history vectors (last 50, decay 0.9). On valid: topic-only = M2 fused with the user's
  category share (`user_cat_affinity`, alpha grid 0 to 1 step 0.1); topic + residual = topic-only fused with the residual match (beta grid,
  same). Family E5 = {(a) overall within-category AUC vs 0.5 on valid, (b) topic+residual - topic-only on valid}. Control: shuffled labels.
- **Charts**: within-category AUC per category; M2 vs topic-only vs topic + residual emotion vs M3.

## E6 Time patterns (weight 5%, exploratory)
- Slots by impression hour: night 0-5, morning 6-11, afternoon 12-17, evening 18-23; weekday vs weekend (fit only: 9-10 Nov are a weekend).
- AUC of M2 and M3 and the gain per slot: fit (scores computed with the frozen settings), valid and dev from the saved files. Clicked-item
  emotion (mean go_pct per emotion) by hour, fit and valid. Exploratory: CIs, no claims. No family.
- **Charts**: gain by slot; clicked emotion by hour heatmap.

## E7 Best fair versions: M3b+ and M5b (weight 25%)
- **M3b+** (choices on valid, in this order, each with the defaults if its rule fails): (1) the E2 window; (2) the same-day blend if the E2
  value rule passed; (3) per-tertile alpha if M3b - M3 on valid (built on 1 and 2) is positive with p < 0.05, else alpha 0.8. Frozen, then dev once.
- **M5b**: LightGBM LambdaRank as M4 (D63), trained on fit, early stopping on valid, 3 seeds (42, 43, 44). Variants:
  - V1 = M4 features + stable personal-match features only: `cong_pct_cos_personal`, `cong_six_negkl_personal`, `cong_valence_absdist_personal`.
  - V2 = V1 + consistency + short-window personal match (the E2 window, or last 10 if the current profile won) + exposure bucket.
  - V3 = V2 + two broad item features: `go_vad_valence`, `go_vad_dominance`.
  - V4 = V2 with stronger regularisation: num_leaves 15, min_child_samples 500, colsample_bytree 0.7, reg_lambda 1.0.
  - V5 = V4 + monotone constraint +1 on every personal-match feature.
  V1 to V3 use M4's frozen hyperparameters (31 leaves, rate 0.05, min child 50). Reference for each = the same features minus emotion,
  same hyperparameters and seeds: M4 (V1), M4 + exposure (V2, V3), M4 + exposure regularised (V4, V5).
- **Choice**: highest mean valid AUC over the 3 seeds; differences under 0.0005 count as a tie, won by the simpler variant
  (order V1, V2, V4, V5, V3). Final M5b on dev = the mean of the 3 seeds' predictions (the valid ensemble is reported too).
- **Shuffled-label controls**: M3b+ scored against labels permuted within impressions (valid and dev, from the files); M5b retrained on fit with
  labels permuted within impressions (valid AUC must be near 0.5).
- **Dev family E7** (AUC, Holm over 8): M3b+ vs M2, M3, M4, M5 and M5b vs M2, M3, M4, M5. All 10 metrics per comparison as before, user-level AUC check.
- **Charts**: valid AUC of V1 to V5 against their references (error bars over seeds); final AUC of M2, M3, M3b+, M4, M5, M5b with CIs;
  M5b feature importance; M3b+ gain by category and by history length.

## E8 Emotional personas (weight 5%)
- User vector = long-term emotion profile (last 50, decay 0.9), users with 5+ known items, standardised with fit means and SDs. k-means
  (k = 3 to 6, 10 starts, seed 42) on fit users; k by the highest silhouette (20,000-user sample). Names from the 2 emotions with the highest
  centroid z-scores. Stability: k-means refit on valid users, clusters matched by centroid distance, adjusted Rand index on valid users.
- Gain of M3 - M2 and M3b+ - M3 per persona, valid and dev from the files. Descriptive, no family.
- **Charts**: 2-D map (PCA), persona sizes, emotion profile per persona, gain per persona.

## E9 Diversity / filter-bubble check (weight 5%)
- Per impression, the top min(10, n) candidates of each model: emotion diversity = number of distinct v2b labels; category diversity =
  number of distinct categories. M2, M3, M3b+, M4, M5b on dev (and valid). Paired t-tests against M2. Descriptive, no family.
- **Chart**: diversity bars with an ethics note.

## Order of work and commits
Phase 0 hygiene and merge; this plan and spec v5 committed; shared code and tests; E1 to E9 (one commit each); findings report.
