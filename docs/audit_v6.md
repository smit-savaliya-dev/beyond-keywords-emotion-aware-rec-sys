# Audit v6: what went wrong before, checked with data (fit and valid only)

Dev numbers this round are a SECOND LOOK; the primary evidence is the rolling-origin folds on days 9-14. Script `scripts/07_00_audit.py`,
tables `results/tables/audit_v6_*.csv`, charts `results/figures/v6_improve/audit_*.png`.

**a) Item overlap: YES, it explains the valid-vs-dev gap of M5b.** 70.8% of valid candidate rows show an article already shown in fit
(46% of unique valid articles; 63% of valid clicks). On dev only 19.4% of rows were shown in fit, but 78.7% were shown in fit or valid.
M5 and M5b were trained on fit only, so the item fingerprints they learned (raw item emotion columns) paid off on valid and not on dev.
A model retrained on days 9-14 would see most dev items again, which is one more reason to keep raw item emotion out of the trees.

**b) Seed noise floor: LARGE for M4.** Five seeds (42-46) on fold F4 (train fit, test valid): M4 AUC 0.6748, SD 0.0023 (range
0.6716-0.6782); M5 0.6882, SD 0.0004. A gain below about 0.002 for M4-type models cannot be told apart from seed noise. M3's dev gain
(+0.0029) and M5b's valid gain over V1 are of this size, so per-seed comparisons are needed.

**c) M4's features: several are missing.** Present: SBERT user-item similarity (m2_score), TF-IDF similarity, item popularity (smoothed CTR
from strictly earlier days), user category share from the history, history length, title length, number of candidates, category and
subcategory. Missing: within-impression normalisation (rank or z-score inside the list), recent CTR over hours (1 h / 6 h / 24 h), the user's
earlier log clicks per category (only the history column is used), article age since first seen, exposure count. The M3b+ score is not in M4
(correct: it is emotion). All missing ones are added to M4v6 (plan v6).

**d) What carries M4's 0.6745 (drop one group, valid, seed 42):** Sentence-BERT match -0.0270, category and subcategory -0.0151,
user category share -0.0070, popularity -0.0060, TF-IDF match -0.0016, context (history, title, list size) +0.0001. Content matching and topic
do most of the work.

**e) Tuning parity: NOT equal before.** M4 and M5 each had the same 8-config grid (equal). But M2 had 4 configs and M3 44 (variants x alphas),
and M5b's five variants had no search at all (M4's settings). From now on every model and its twin get the same Optuna budget (8 trials).

**f) Valid day vs dev day:** valid lists are longer (mean 40.4 candidates vs 37.5 on dev and 36.5 on fit; median 27 vs 23), histories are
similar (33.1 vs 32.3 items), cold start is 2.6% vs 3.0%. Longer lists and the high overlap make valid easier for item-memorising models.

**g) What M5 used:** of M5's mean absolute contribution, 18.4% came from 13 raw item emotion columns (go_vad_dominance alone 8.5%) and
3.6% from 4 user-relative congruence columns; 78.0% from the 9 non-emotion features. M5's emotion signal was mostly item identity.

**Our mistakes, in short:** we read valid gains that came from item overlap as emotion effects (M5, M5b); we compared models with unequal
tuning (M2 vs M3, M5b); we ignored a seed-noise floor of about 0.002; and M4 lacked standard time-causal features, so a weak twin made
emotion look more useful than it is.
