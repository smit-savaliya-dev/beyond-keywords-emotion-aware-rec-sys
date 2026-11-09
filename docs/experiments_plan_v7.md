# Experiments plan v7: same-day audit and two regimes (pre-declared)

Committed before any v7 result (decisions D90-D93). Code `src/same_day.py`, `scripts/08_01_regimes.py`, `scripts/08_02_regime_analysis.py`.
Anything added after a result is labelled **POST-HOC EXPLORATORY** and is not used for the headline claim.

**Honesty note.** Dev (15 Nov) was already seen in weeks 7 and 8. Dev numbers here are a further look; the primary evidence is the
rolling-origin folds on days 9-14 (F1-F4). No existing model is rerun on dev; only the two new final R1 models get one dev run each.

## Why
M4v6 (plan v6) beat the old M4 by +0.037 AUC on dev. Its new columns count clicks and impressions strictly before the scored impression,
which includes earlier impressions **of the same day**. On the final test day that means earlier dev-day labels. This is time-causal (an
online setting, D75), but it is not the standard offline protocol, where a scored day sees nothing from itself.

## Two regimes (declared now, not changeable afterwards)
- **R1 STANDARD (primary).** Every behavioural feature uses data from previous days only: for a scored day, no click or impression from that
  same day. The thesis claim is made in R1.
- **R2 REAL-TIME (secondary).** Same-day earlier clicks and impressions are allowed, exactly as M4v6 / M5v6 of plan v6.

Same-day columns of M4v6 (`src/same_day.CATALOGUE`): ctr_1h/6h/24h, shows_1h/6h/24h, exposure_log, age_hours, user_log_clicks,
user_log_cat_share, ctr_24h_rank, ctr_24h_z. Every other M4v6 / M5v6 input (old M4 columns, M1/M2 scores, history column, item text, emotion
columns, M3b+ score) uses no same-day event. NRMS (M6/M7) uses history and text only, so it is the same model in both regimes.

R1 replacements (`pd_` columns): the same quantities computed from events before 00:00 of the row's day: item CTR and shows in the last
1 h / 6 h / 24 h before midnight, all earlier shows, item age at midnight (missing if never shown before the day), the user's logged clicks
before the day and their share in the candidate's category, and the within-list rank and z-score of pd_ctr_24h. Leakage test: changing or
adding any same-day event never changes a row (`tests/test_same_day.py`).

## Models (LightGBM LambdaRank, rolling folds as plan v6)
- **T1** old M4 columns only (the 9 M4 features).
- **T2** M4v6 full (R2) - exists (plan v6).
- **T3 = M4v6_R1** (R1 twin): M4 columns + pd_ columns + within-list rank/z of pop_ctr, m1_score, m2_score, user_cat_affinity, pd_ctr_24h.
- **T4 = M5v6_R1 variants** (R1 emotion; max 3, all user-relative, no raw item-level emotion column):
  F2 (personal match family + consistency + consistency x personal + persona), F3 (F2 + residual-topic congruence), F6 (F2 + M3b+ score).
  Nothing else is added after seeing results.
- **T5** R2 emotion twin pair: M4v6 vs M5v6 (= F2), reused from plan v6; no new variant.

**Tuning parity.** Each new model (T1, T3, F2/F3/F6 in R1) gets the plan-v6 budget: Optuna TPE seed 42, 8 trials, plan-v6 space (not the
strong F4 space), objective = mean AUC of seed 42 on F3 and F4; then the best setting x seeds 42/43/44 x folds F1-F4.

**Rule 8 in R1.** A variant is eligible only if its seed-mean AUC beats T3 in at least 2 of the 3 folds F2-F4 AND its mean gain over
F1-F4 exceeds T3's seed noise (SD across seeds, averaged over folds). Highest mean F1-F4 AUC among eligible wins; ties within 0.0005 go to the
simpler (order F2, F6, F3). If none is eligible the declared R1 emotion model is F2 and every report says "did not pass the rule".

**Controls.** Shuffled-label retrain (labels permuted within impressions, seed 42) on F4 with a FIXED number of rounds (the model's mean best
iteration), for T3, the R1 emotion model, T2 and T5 (fixed rounds avoid the early-stopping selection bias of D83). Valid AUC must be near 0.5.
Dev shuffled-label check on every new dev file.

## Dev (one run per new final model)
M4v6_R1 and the selected M5v6_R1 are frozen (D-entries with the frozen hash) before their single dev run: retrained on days 9-14 with frozen
settings, rounds = mean best iteration over F1-F4 per seed, 3-seed mean, exactly as plan v6. Dev R1 features use days up to 14 Nov only.

## Blends in R1
B0_R1 = w z(M4v6_R1) + (1 - w) z(M6); B1_R1 = w z(M5v6_R1) + (1 - w) z(M7); w in 0, 0.1, ..., 1 chosen on valid (F4 predictions);
dev from the saved dev files (no extra dev run). B1 - B0 is reported in R1 and R2, and, as in D89, also at the same weight (labelled post-hoc
in R2, declared here for R1).

## Questions and statistics
- **Q1 (survival).** Share of M4v6's gain over old M4 that survives in R1: folds (T3 - T1) / (T2 - T1), mean over F1-F4; dev
  (M4v6_R1 - M4) / (M4v6 - M4) with M4 the existing old-M4 dev file. Reported as a number.
- **Q2 (emotion vs twin).** For each regime: twin AUC, emotion AUC, gain, 95% CI (user-level bootstrap), paired t-test p, Holm p, dz,
  per-fold gains, seed-noise SD, folds won, Rule 8 verdict.
- **Dev families (Holm within each).** R1 primary: M5v6_R1 - M4v6_R1, B1_R1 - B0_R1, M7 - M6, M7 - M7ctrl.
  Secondary: M4v6_R1 - M4, M4v6 (R2) - M4v6_R1, B0_R1 - M4v6_R1. The R2 primary family of plan v6 is reported unchanged.

## Step 2: where emotion has the best chance (exploratory, declared segments)
M5v6_R1 vs M4v6_R1 on dev (and F4/valid): consistency tertile (E1 fit cut-offs), persona (E8), clicked category (top categories), known
history items (plan v6 bins), time slot. All cells reported; Holm over all declared cells together. Questions: does gain rise with consistency?
Is the joy-reader loss absent? Is the gain larger in R1 than in R2 (same segments for M5v6 - M4v6)?

## Step 3: error analysis (descriptive)
Per-impression dev AUC of M5v6_R1 minus M4v6_R1, among impressions with at least 10 candidates and at least one click: the 10 largest gains and
10 largest losses, with clicked headline, category, its top emotions, the personal match of the clicked item and the user's consistency.
