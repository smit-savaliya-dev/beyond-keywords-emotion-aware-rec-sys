# Plan v8: improving the neural ranker + emotion (M7) — pre-declared

Written and committed on 2026-11-16 BEFORE any v8 number was computed (D105).

## Why
In plan v6 no NRMS emotion variant passed Rule 8 (valid: M7b - M6 = +0.0002 AUC, M6 seed SD 0.0018), and the final test
gain is +0.0002 (not significant). Diagnosis from the saved checkpoints, before any new run:
- M7b adds `g(user) * w * cos(profile, item)` with the PLAIN cosine. On go_pct vectors this cosine varies little between
  candidates, while the NRMS dot-product score varies a lot, so the term is tiny next to it (learned w is about 0.4).
- The stronger signal from the analysis, the PERSONAL match (cos to the profile minus cos to the corpus mean, dz 0.25 on fit,
  0.24 on valid; it is the M3 score), is not used.
- The emotion parameters learn at the same rate (1e-4) as the whole network and start at w = 0.

## Variants (all trained on fit, days 9-13; early stopping and every choice on valid, 14 Nov; dev is NOT used)
| Code | What | Training |
|---|---|---|
| M7d | Late fusion, the M3 recipe on top of M6: `a * minmax(M6) + (1 - a) * minmax(personal match)` within each impression | none (uses M6 checkpoints) |
| M7e | NRMS + trained personal-match head: `dot(user, item) + (w0 + w1 * consistency_z) * z(personal match)`; z = standardised with fit mean and SD; w0 starts at 1, emotion parameters learn at 1e-2 | 3 seeds, as M6 |
| M7f | M7e plus the M7a item-emotion projection, with the projection also at 1e-2 | 3 seeds, as M6 |

M7d alpha grid: 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5 (1.0 = M6). To avoid choosing and scoring on the same rows,
M7d's valid number is CROSS-FITTED: valid users are split in two halves (seeded permutation, SEED = 42); the alpha chosen on
one half is applied to the other. The alpha frozen for any later dev run is the best alpha on all of valid.

## Selection rule (same as plan v6 Rule 8, seeds replace folds)
- Per seed s in {42, 43, 44}: gain = variant(s) - M6(s) in valid AUC. For M7d the variant is the fusion applied to M6 seed s.
- Eligible if the mean gain > the M6 seed SD (0.0018, from v6_nrms_seeds_valid.csv) AND at least 2 of 3 seeds win.
- Among eligible variants: highest mean valid AUC; ties within 0.0005 go to the simpler one (M7d < M7e < M7f).
- Control: the selected variant rebuilt with item emotions permuted across items (the M7ctrl permutation) must score lower
  on valid than the selected variant; otherwise the gain is not called an emotion effect.
- If nothing is eligible, the report says so and M7b stays the final neural + emotion model.
- Primary metric AUC (as all plans); nDCG@10, MRR and the other metrics are reported for every variant.

## Dev
The final test day has already been seen several times. A dev run of the selected variant is made only after the student
approves it explicitly, once, with its frozen marker recorded first. Until then every v8 number is a validation number and is
labelled so.
