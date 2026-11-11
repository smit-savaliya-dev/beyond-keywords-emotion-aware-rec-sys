# Experiments plan v6: fair twins, NRMS and blends (pre-declared)

Committed before any week-8 model result. The Stage 0 audit (`docs/audit_v6.md`) ran first; it trains no new model and only informed which
no-emotion features M4v6 gets. Decisions D79 to D84. Anything added later is marked **post-hoc exploratory**.

**Honesty note (repeated in every report).** Dev (15 Nov) was already looked at in week 7, and that motivated some of these changes.
Dev numbers this round are a SECOND LOOK; the primary evidence is the rolling-origin folds on days 9-14.

## Rules
- **Fair test.** The emotion claim is always "model WITH emotion minus the SAME model WITHOUT emotion": same other features, same tuning
  budget, same seeds, same training rows. "Best overall model" is reported separately.
- **No raw item-level emotion columns** in M4v6 or M5v6 (they act as item IDs, D69-D71, D78). Item emotion enters only inside NRMS (M7) or as
  user-relative features.
- **Time-causal.** History column, item text, or events strictly earlier than the impression. Earlier-impression features on valid and dev
  use labels of strictly earlier impressions of the same day: an online setting, as D75. Every new feature has a leakage test.
- **Dev once** per new final model (M4v6, M5v6, M6, M7, M7ctrl); blends B0/B1 are computed from those dev files (no extra dev run).
- **Statistics.** Per-impression paired t-test on AUC (primary), 95% CI, dz, user-level check, Holm over the family below, shuffled-label control.

## Stage 1: rolling-origin folds
F1 train 9-10 Nov, test 11; F2 train 9-11, test 12; F3 train 9-12, test 13; F4 train 9-13, test 14 (= valid). Early stopping uses the fold's
test day (as D66 did with valid), so fold numbers are slightly optimistic for every model alike. Final tree models are retrained on days 9-14
with the number of rounds = the mean best iteration over F1-F4 for that seed; dev is scored once with the mean of 3 seeds.

## Stage 2: M4v6 and M5v6 (LightGBM LambdaRank, as D63 otherwise)
- **M4v6 features (no emotion)** = M4's 9 features + recent item CTR in the last 1 h / 6 h / 24 h and the log of the shows behind each
  (strictly earlier, smoothed with a = 10 toward the fixed prior 0.04) + log exposure (earlier impressions) + article age in hours since its
  first impression + the user's earlier log clicks (log count) and their share in the candidate's category + within-impression rank and z-score of
  pop_ctr, m1_score, m2_score, user_cat_affinity and ctr_24h. SBERT similarity (m2_score), history length and list size are already in M4.
- **M5v6 variants** (max 6; all = M4v6 features + user-relative emotion):
  - F1: congruence cosine on go_pct, -KL on the six classes, -|valence distance|, each raw, personal and user-specific (9 columns).
  - F2: F1 + consistency (E1) + consistency x personal cosine + persona id (E8 k-means refit on fit users, categorical).
  - F3: F2 + residual-topic congruence (E5: item vector minus its category mean over fit items, cosine with the residual profile).
  - F4: F3 + 28 products profile_i x item_i of the GoEmotions raw scores (profile = last 50, decay 0.9), stronger regularisation space.
  - F5: M3b+ score only (the week-7 late-fusion score, computed per part).
  - F6: F2 + F5.
- **Tuning parity.** Optuna TPE (seed 42), 8 trials per model, the same space for all (num_leaves 15-127, learning_rate 0.02-0.15,
  min_child_samples 20-500, colsample_bytree 0.5-1, reg_lambda 0.001-10; F4: min_child_samples 300-1000, colsample 0.4-0.8, reg_lambda 1-20).
  A trial = seed 42 on F3 and F4, objective = mean AUC. The best setting is then run with seeds 42, 43, 44 on F1-F4.
- **Rule 8 (eligibility).** A variant is eligible only if its seed-mean AUC beats M4v6's in at least 2 of the 3 folds F2-F4 (F1 trains on only
  2 days) AND its mean gain over F1-F4 exceeds the twin's seed noise (SD across seeds of M4v6's AUC, averaged over folds). Among eligible variants
  the highest mean F1-F4 AUC wins; ties within 0.0005 go to the simpler (order F5, F1, F2, F6, F3, F4). If none is eligible, the declared final
  M5v6 is F5 (the late-fusion score) and the report says "emotion did not pass the rule".
- **Controls.** Every variant retrained on F4 with labels permuted within impressions (seed 42); valid AUC must be near 0.5.

## Stage 3: NRMS M6
As Wu et al. (2019): word embeddings -> dropout 0.2 -> MHSA 16 x 16 -> additive attention (query 200) -> 256-d news vector; user = MHSA over
the last 50 history vectors -> additive attention; score = dot. One click + 4 non-clicks from the same impression, softmax cross-entropy;
Adam 1e-4, gradient clip 1.0, AMP, title 30 tokens, 32 impressions per batch, at most 8 epochs, early stopping on valid AUC with patience 1,
seeds 42/43/44. Word embeddings: GloVe could not be downloaded here, so the input-embedding matrix of the cached all-mpnet-base-v2 (with its
tokenizer, lower-cased titles) initialises a trainable embedding (D80). Reproduction gate: M6 valid AUC near 0.63-0.66; below 0.62 the debug
checklist of the task is followed and logged.
- **Budget** (measured: 1% of an epoch = 2-3 s, so about 4-5 min per epoch; scoring valid 20 s). 4 models x 3 seeds x at most 8 epochs, plus
  M7ctrl x 3 seeds: at most about 10 GPU hours, in parallel with about 4.5 CPU hours of trees. Declared reductions: NRMS runs only on fold F4
  (train fit, early-stop on valid; F3 is not run) and the dev run uses these fit-trained checkpoints (no retrain on days 9-14).

## Stage 4: M7 (max 3 variants + control)
Item emotion input = 28 GoEmotions raw scores + 3 VAD values, standardised with fit-item statistics.
- M7a: news vector + W_e e (linear 31 -> 256, small init), on candidates and history.
- M7b: score + g(user) * w * cos(user go_pct profile (last 50, decay 0.9), item go_pct); g = sigmoid(MLP[consistency, has-consistency,
  log(1 + history)]); w learnable (starts at 0).
- M7c: M7a + M7b.
- M7ctrl (for the selected variant only): identical, with item emotion vectors permuted across items by a fixed seed; profile and consistency
  are rebuilt from the permuted vectors.
- **Rule 8 for NRMS** (one fold, so seeds replace folds): eligible if it beats M6 in at least 2 of the 3 seeds (paired by seed) AND the mean gain
  exceeds the SD of M6's seed AUCs. The eligible variant with the highest mean valid AUC is selected (ties within 0.0005: M7b, M7a, M7c).
  If none is eligible, the declared M7 is M7b (the score-level fusion). The emotion-information claim also needs M7 > M7ctrl on valid.

## Stage 5: blends
B0 = w z(M4v6) + (1 - w) z(M6), B1 = w z(M5v6) + (1 - w) z(M7), z-scores within each impression, w in 0, 0.1, ..., 1 chosen on valid (fold F4
predictions). B1b = B1 + v z(personal match) with v in {0, 0.05, 0.1, 0.2, 0.3}, kept only if it beats B1 on valid with p < 0.05.

## Statistical families (dev, Holm within each)
- **Primary (emotion vs twin)**: M5v6 - M4v6, M7 - M6, M7 - M7ctrl, B1 - B0.
- **Secondary (best overall)**: M4v6 - M4, B0 - M4v6, B1 - M5, best of all vs M4.
The same comparisons are reported on the folds (mean over seeds) as the primary evidence.

## Stage 6: analysis of the final emotion models
For M5v6 vs M4v6, M7 vs M6 and B1 vs B0 on dev (and F4/valid): gain by consistency tertile (E1 cut-offs), clicked category, history length,
persona (E8), time slot (exploratory). Questions: does the gain still rise with consistency? Does the joy-reader loss disappear with the gate (M7b)?

## Stage 7: charts
Old v5 charts copied to `archive/figures/archive_v5/`; affected charts regenerated; new charts in `archive/figures/v6_improve/` with a CSV each.
