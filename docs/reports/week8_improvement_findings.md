# Week 8: fair twins, NRMS and blends (plan v6)

**Honesty note.** Dev (15 Nov) was already looked at in week 7, and that motivated some of these changes. Dev numbers this round are a
SECOND LOOK; the primary evidence is the rolling-origin folds on days 9-14 (and, for NRMS, the three seeds on valid). Branch `week8-improve`,
local commits only. Plan `docs/experiments_plan_v6.md` (committed before any result), audit `docs/audit_v6.md`, decisions D79-D89.
Charts `archive/figures/v6_improve/` (`overview_v6.png`), tables `results/tables/v6_*.csv` and `audit_v6_*.csv`. 261 tests pass.

## Per stage
**Stage 0 Audit** - FOUND THE MISTAKES
- Asked: why did M5 and M5b look good on valid and fail on dev?
- Found: 70.8% of valid candidates were already shown in fit but only 19.4% of dev candidates; 18.4% of M5's contribution came from raw item
  emotion columns (item IDs in disguise); M4's seed noise is SD 0.0023, as big as the effects we were reading; tuning was unequal (M2 4 settings vs M3 44).
- Verdict: the earlier emotion gains for trees were partly item memorising and partly noise.

**Stage 1-2 Fair tree models** - M4v6 HELPS a lot (no emotion); emotion TOO SMALL
- Asked: with a stronger, fairly tuned no-emotion model, does emotion still add?
- Found: the time-causal features (recent CTR, article age, the user's earlier category clicks, within-list ranks) lift the tree from 0.6739 to
  0.7111 on the final test (+0.0372). The selected emotion twin M5v6 (variant F2) adds +0.0012 on the folds (F1 +0.0034, F2 +0.0006,
  F3 -0.0007, F4 +0.0005) and +0.0007 on the final test (95% CI 0.0001-0.0014, Holm p 0.09, user-level p 0.07).
- Verdict: emotion passed the eligibility rule by a hair, but the final-test gain is not significant after Holm: TOO SMALL.

**Stage 3 NRMS M6** - REPRODUCED
- Asked: does our NRMS work?
- Found: valid AUC 0.6387 (3 seeds 0.6375-0.6408), inside the 0.63-0.66 target; 4-5 epochs, early stopping as declared. Final test 0.6082:
  NRMS drops more from valid to dev than the trees, like M2 did.
- Verdict: working, but weaker than the tree models here.

**Stage 4 NRMS + emotion M7** - NO EFFECT
- Asked: does emotion inside NRMS help, and is it the emotion information or just extra capacity?
- Found: no variant beat M6 by more than its seed noise (M7a +0.0014, M7c +0.0013, M7b +0.0002; SD 0.0018), so the declared fallback M7b
  is M7. Final test: M7 - M6 +0.0002 (p 0.33); M7 - M7ctrl (emotion permuted) -0.0006 (Holm p 0.14).
- Verdict: no sign that emotion information helps NRMS.

**Stage 5 Blends** - best overall B1, but the emotion part is TOO SMALL
- Asked: does a tree + NRMS blend beat the trees, and does emotion help at system level?
- Found: B0 0.7075, B1 0.7120 on the final test. The pre-declared B1 - B0 = +0.0045 (Holm p 6e-34) is mostly a weight difference
  (B1 uses tree weight 0.8, B0 0.7, and NRMS is weak on dev). Post-hoc at the same weight: +0.0007 to +0.0008, the tree-level gain.
- Verdict: the blend does not beat M4v6 alone without emotion (B0 - M4v6 -0.0036); the emotion part stays tiny.

**Stage 6 Where emotion still helps** (descriptive)
- Asked: does the week-7 pattern (more gain for consistent users; loss for joy readers) survive against fair twins?
- Found: the direction survives but is small: M5v6 - M4v6 low/mid/high consistency +0.0005/+0.0008/+0.0010, M7 - M6 +0.0006/+0.0002/+0.0008
  (high significant). The joy-reader loss is gone: M5v6 +0.0023 and gated M7 +0.0005 for "excitement + joy" readers.
- Verdict: a weak, consistent pattern, not a usable gain.

## Did emotion beat its twin?
| Comparison | Folds / valid (primary) | Final test (second look) | Holm p | dz | Verdict |
|---|---|---|---|---|---|
| M5v6 - M4v6 (trees) | +0.0012 (mean of 4 folds) | +0.0007 (0.0001 to 0.0014) | 0.093 | 0.008 | TOO SMALL |
| M7 - M6 (NRMS) | +0.0003 (valid, 3 seeds) | +0.0002 (-0.0002 to 0.0007) | 0.33 | 0.004 | NO EFFECT |
| M7 - M7ctrl (emotion information) | +0.0003 (valid) | -0.0006 (-0.0013 to 0.0001) | 0.14 | -0.007 | NO EFFECT |
| B1 - B0 (blends) | +0.0004 (valid) | +0.0045 (+0.0008 at the same weight, post-hoc) | 6e-34 | 0.045 | TOO SMALL (confounded) |

Shuffled-label controls on the final test give AUC 0.499-0.501 for every emotion model.

## Best overall model
B1 (M5v6 + M7 blend) 0.7120, statistically tied with M5v6 0.7118 and M4v6 0.7111; all three beat the week-6 M4 (0.6739) by +0.037 to +0.038.
The big step comes from better time-causal features, not from emotion.

## What to claim in the thesis and what NOT to claim
- Claim: a fair test (same model with and without emotion, same tuning budget, same seeds) finds at most a tiny emotion gain: +0.0007 AUC for
  the tree model on the final test (not significant after Holm) and none for NRMS.
- Claim: earlier apparent gains (M5, M5b) were item memorising on a validation day that shared 71% of its items with training, and the audit
  shows how this was found and fixed.
- Claim: emotional context helps slightly more for emotionally consistent users, a weak but repeated pattern; descriptive only.
- Claim: the strongest model is a feature-rich, time-causal LightGBM ranker; emotion is not needed to reach it.
- Do NOT claim that emotion improves news (or ad) ranking, that B1's +0.0045 is an emotion effect, or anything about "the user's emotional state".

## Limitations
- Dev is a second look (D79); the folds are the primary evidence, and early stopping uses each fold's test day, which is mildly optimistic for every model alike.
- GoEmotions was trained on Reddit comments; news headlines and abstracts are a domain shift (SemEval validation covers headlines only).
- MIND has no ads: any claim about ad ranking is a transfer argument only.
- NRMS used mpnet input embeddings instead of GloVe (not downloadable here, D80), ran on fold F4 only, and was not retrained on days 9-14.
- Tree shuffled-label retrains are above 0.5 (0.52-0.57) because of early-stopping selection on true labels; they show no emotion advantage.
- Compute incident (D82): parallel runs filled the C: drive; jobs were rerun from checkpoints, no result changed.
