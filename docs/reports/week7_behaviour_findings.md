# Week 7: when does emotional context help ranking? (plan v5)

Branch `week7-behaviour`, local commits only. All experiments were pre-declared in `docs/experiments_plan_v5.md` (committed before any
result). Fit chose cut-offs, valid chose settings, dev was used once each for the two new final models (M3b+, M5b); all other dev numbers
come from the saved dev prediction files. Numbers are AUC differences unless stated. Charts: `archive/figures/v5_behaviour/`
(`overview.png` is the one-page summary). Tables: `results/tables/behaviour_*.csv`. 250 tests pass.

## Results per experiment

**E1 Consistent vs varied users** - HELPS (for consistent users)
- Asked: does emotion help users whose history has a steady emotional tone more than users who read across moods?
- Found: on the final test the gain of M3 over M2 rises from +0.0013 (most varied third) to +0.0053 (most consistent third); high minus low
  +0.0040 (Holm p 7e-7), the same direction on valid (+0.0036, Holm p 0.002). Controls (consistency permuted, labels shuffled) are flat.
- But: choosing the fusion weight per group (M3b) did not beat one weight for everyone (+0.0003, p 0.10).

**E2 Short-window and same-day profiles** - NO EFFECT
- Asked: is recent reading (last 5 to 20 items, or clicks earlier the same day) a better emotional context than the long history?
- Found: shorter windows are worse (last 5: +0.0011 over M2 against +0.0035 for 50 items, valid). A third of impressions have an earlier
  same-day click, but blending it in adds nothing (+0.00001 on those impressions, p 0.98). The best profile is all 50 items without decay.

**E3 Emotion drift** - NO EFFECT
- Asked: do users whose recent emotions differ from their past benefit from a short profile?
- Found: the short profile is worse in every drift group (-0.0022 to -0.0033, valid); drift does not change that (rho -0.005, p 0.51; dev
  M3b+ - M3 high minus low -0.0004, p 0.49).

**E4 New articles** - TOO SMALL
- Asked: does emotion help most for articles nobody has clicked yet?
- Found: only 0.1-0.3% of candidates are never-shown articles, so the "new" group has 49 to 70 usable impressions (flagged too small).
  Between rarely and often shown articles the gain is the same (dev +0.0026 vs +0.0033).

**E5 Emotion within the same topic** - HELPS (emotion is not just topic)
- Asked: is emotion only a disguised topic signal?
- Found: among candidates of the same category, the emotion match alone ranks clicks above chance (AUC 0.555, 95% CI 0.551-0.559, valid;
  0.554 on fit). Emotion with the category mean removed still adds +0.0032 on top of M2 plus topic (valid, Holm p < 1e-5; chosen on valid,
  so optimistic). On the final test M3 beats M2 by +0.0040 within categories.

**E6 Time of day** (exploratory) - NO CLAIM
- Asked: does the gain depend on when people read?
- Found: it is positive in every slot and largest at night on all three parts (dev night +0.0062, morning +0.0018). One week of data,
  no correction: a pattern to test later, not a result.

**E7 Best fair versions** - M3b+ HELPS a little; M5b HURTS
- Asked: does combining the findings give a better emotion model?
- Found: M3b+ kept only one change (50 items without decay; the same-day blend and the adaptive weight failed their rules). Final test
  0.6293 vs M3 0.6282: +0.0011 (95% CI 0.0008-0.0014, Holm p 3e-12), +0.0040 over M2. M5b (variant V3, chosen on valid where it gained
  +0.0093 over M4) scores 0.6647 on the final test: -0.0092 below M4 (Holm p 3e-34). This was predicted before the dev run (D78): the only
  variant that gained on valid uses item-level emotion scores, which act as item IDs on a day that shares 71% of its items with training.
  Shuffled-label controls on the final test give AUC 0.5015 and 0.5014.

**E8 Emotional personas** - HELPS for some readers, HURTS for others
- Asked: are there reader types for whom emotion matters more?
- Found: three personas, stable on valid (adjusted Rand 0.94) but weakly separated (silhouette 0.19). Emotion helps "fear + nervousness"
  readers most (dev +0.0084) and "annoyance + optimism" readers (+0.0044), but slightly hurts "excitement + joy" readers (-0.0028; valid -0.0021).
  M3b+ removes most of that loss for joy readers (+0.0026 over M3).

**E9 Diversity / filter bubble** - TOO SMALL to matter
- Asked: does matching emotions narrow what people see?
- Found: M3 shows 0.028 fewer distinct emotions in the top 10 (3.95 to 3.92, under 1%), M3b+ 0.022; topic variety does not change
  (+0.004, p 0.07). Small, but a ranking that narrows emotional variety can reinforce mood loops, so it should be monitored.

## When does emotion help, when does it not, and why
1. Emotion helps on top of a content model (M3 over M2) and helps more for users with a steady emotional tone and for readers of anxious or
   negative news ("fear + nervousness"): their history describes what they will click next.
2. It does not come from recency: the long history is the best emotional context, same-day clicks and short windows add nothing.
3. It is not topic in disguise: it ranks inside a category and adds beyond topic.
4. It does not help a strong learned ranker. M5 and now M5b lose to M4 on the final test. Inside LightGBM, item-level emotion scores become an
   item fingerprint that pays off only when the test items were seen in training, which is not the case on the final test day.
5. It can hurt readers of upbeat news ("excitement + joy"), so a single global weight is a compromise.

## What to claim in the thesis and what NOT to claim
- Claim: emotional context inferred from a user's recent reading gives a small, replicated gain over a content-only model (M3 +0.0029,
  M3b+ +0.0040 over M2 on the final test), larger for emotionally consistent users, and not explained by topic.
- Claim: the best ranker without emotion (M4) is not beaten by any emotion version; emotion is not a production-level improvement here.
- Claim, with care: the persona and time-of-day patterns are descriptive (personas weakly separated; time exploratory).
- Do NOT claim that emotion improves ranking in general, that recent mood matters more than long-term taste, that emotion helps for new
  articles (not testable here), or anything about "the user's emotional state".
- Do NOT read the valid gains of M3b, M5b or the E5 fusion as results: valid chose them, and M5b shows how far valid can mislead.
