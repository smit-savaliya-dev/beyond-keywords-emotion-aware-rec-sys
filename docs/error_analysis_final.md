# Error analysis: R1 emotion model (M5v6_R1) against its twin (M4v6_R1)

Final test (dev), read from the saved prediction files (a further look at dev). Impressions with at least 10 candidates
(57,228 impressions). Per impression: AUC of the emotion model minus AUC of the twin. Median gain +0.0000;
36.3% of impressions improve, 31.6% get worse, 32.0% unchanged.
Columns: the first clicked headline, its two strongest emotions (GoEmotions percentile among all articles, 0-1), its personal match (how much closer it is to the user's
emotional history than to the average user's; rank 1 = best match in the list), the user's consistency (how similar their history items are
emotionally, 0-1), known history items, and where each model put the clicked story (1 = top).

## Summary
1. When emotion helped most, the clicked story was among the best emotional matches for that reader (personal-match rank 1-5), so the emotion model moved it up (e.g. from 10th to 3rd).
2. Most big wins are strongly fearful news (a school shooting, a fatal stabbing) clicked by readers whose history is also fear-heavy: emotion works when a reader's taste is clear and the story's tone is strong.
3. When emotion hurt most, the reader clicked something unlike their emotional history (match rank near the bottom), often with very short histories (1-3 items) where the profile is guesswork.
4. A few stories dominate both lists (the Kaepernick opinion piece appears 6 times, helped twice and hurt four times): the same story's emotion pushes it up for some readers and down for others.
5. These are extremes. Across all impressions about as many get worse as better (median change 0), and the average gain is small; the examples show how emotion works, not that it usually helps.

## 10 impressions where emotion helped most
| # | Clicked headline | Category | Top emotions | Personal match (rank in list) | Consistency | History | Rank twin -> emotion | AUC gain |
|---|---|---|---|---|---|---|---|---|
| 1 | South Carolina teen gets life in prison for deadly elementary school shooting | news | fear 0.98, nervousness 0.87 | +0.042 (1 of 12) | 1.00 | 2 | 10 -> 3 | +0.636 |
| 2 | This stately home is having the ultimate yard sale | finance | excitement 0.91, desire 0.88 | +0.106 (1 of 16) | 0.76 | 50 | 13 -> 5 | +0.533 |
| 3 | South Carolina teen gets life in prison for deadly elementary school shooting | news | fear 0.98, nervousness 0.87 | +0.047 (3 of 18) | 0.92 | 4 | 13 -> 4 | +0.529 |
| 4 | Former North Carolina State, NBA player Anthony Grundy dies in stabbing, police say | sports | fear 0.97, nervousness 0.90 | +0.016 (5 of 42) | 0.85 | 30 | 29 -> 8 | +0.512 |
| 5 | Opinion: Colin Kaepernick is about to get what he deserves: a chance | sports | desire 0.96, optimism 0.95 | +0.035 (1 of 11) | 0.86 | 3 | 6 -> 1 | +0.500 |
| 6 | Selena Gomez Stands Up for Taylor Swift Amid Her Music Battle: 'It's Greed, Manipulation and Power' | music | nervousness 0.88, fear 0.75 | +0.041 (5 of 25) | 0.89 | 4 | 23 -> 11 | +0.500 |
| 7 | Opinion: Colin Kaepernick is about to get what he deserves: a chance | sports | desire 0.96, optimism 0.95 | -0.087 (11 of 11) | n/a | 1 | 8 -> 3 | +0.500 |
| 8 | 30 Best Black Friday Deals from Costco | lifestyle | joy 0.93, excitement 0.90 | +0.047 (1 of 11) | 0.79 | 4 | 9 -> 4 | +0.500 |
| 9 | South Carolina teen gets life in prison for deadly elementary school shooting | news | fear 0.98, nervousness 0.87 | +0.044 (2 of 20) | 0.84 | 5 | 18 -> 9 | +0.474 |
| 10 | Former North Carolina State, NBA player Anthony Grundy dies in stabbing, police say | sports | fear 0.97, nervousness 0.90 | +0.026 (3 of 12) | 0.78 | 13 | 9 -> 4 | +0.455 |

## 10 impressions where emotion hurt most
| # | Clicked headline | Category | Top emotions | Personal match (rank in list) | Consistency | History | Rank twin -> emotion | AUC gain |
|---|---|---|---|---|---|---|---|---|
| 1 | Opinion: Colin Kaepernick is about to get what he deserves: a chance | sports | desire 0.96, optimism 0.95 | -0.042 (7 of 10) | 0.84 | 15 | 4 -> 9 | -0.556 |
| 2 | The world's most profitable company is about to go public | finance | annoyance 0.92, optimism 0.90 | -0.196 (15 of 15) | 1.00 | 2 | 8 -> 15 | -0.500 |
| 3 | Taylor Swift Says Scooter Braun, Scott Borchetta Are Blocking Her From Playing Old Hits at AMAs | music | annoyance 0.72, joy 0.40 | -0.242 (18 of 20) | n/a | 1 | 4 -> 13 | -0.474 |
| 4 | Judge agrees Alabama Islamic State recruit is not US citizen | news | optimism 0.71, curiosity 0.50 | -0.012 (5 of 10) | n/a | 1 | 4 -> 8 | -0.444 |
| 5 | Opinion: Colin Kaepernick is about to get what he deserves: a chance | sports | desire 0.96, optimism 0.95 | -0.111 (9 of 10) | 0.79 | 6 | 1 -> 5 | -0.444 |
| 6 | The Day in Sports: Thursday, November 14, 2019 | sports | excitement 0.94, desire 0.91 | -0.182 (22 of 28) | 0.77 | 2 | 10 -> 21 | -0.407 |
| 7 | Opinion: Colin Kaepernick is about to get what he deserves: a chance | sports | desire 0.96, optimism 0.95 | -0.037 (10 of 11) | 0.81 | 18 | 1 -> 5 | -0.400 |
| 8 | Opinion: Colin Kaepernick is about to get what he deserves: a chance | sports | desire 0.96, optimism 0.95 | -0.045 (9 of 11) | 0.70 | 4 | 6 -> 10 | -0.400 |
| 9 | White House releases summary of Trump's first call with Ukraine's Zelensky | news | excitement 0.58, joy 0.53 | -0.087 (14 of 16) | 0.80 | 9 | 2 -> 8 | -0.400 |
| 10 | Taylor Swift Rep Hits Back at Big Machine, Claims She's Actually Owed $7.9 Million in Unpaid Royalties | music | annoyance 0.83, joy 0.32 | -0.037 (6 of 11) | 0.74 | 3 | 2 -> 6 | -0.400 |

Chart: `results/figures/v7_final/error_analysis.png`. Table: `results/tables/v7_error_analysis.csv`.
