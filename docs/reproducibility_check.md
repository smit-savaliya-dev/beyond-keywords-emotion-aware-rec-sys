# Clean-clone reproducibility check (2026-11-09)

1. `git clone` of this repository into a new folder (`beyond-keywords-clone`, commit f0adbed at clone time; tests rerun on the final commit).
2. Fresh venv with Python 3.11.9, `pip install -r requirements.txt` from the internet (PyPI was reachable; pip cache redirected to D:).
   Newer versions were installed than in the project venv: pandas 3.1.1 (3.0.6), pyarrow 26.0.1 (25.0.1), torch 2.15.0 CPU (2.11.0+cu128),
   transformers 5.22.2 (5.18.0); lightgbm 4.7.0, numpy 2.4.6, sentence-transformers 6.2.0 and optuna 5.0.0 are equal.
   `requirements-lock.txt` now records the exact project versions.
3. Raw MIND-small is not in git (licence; data is not redistributed): `data/raw/train` and `data/raw/dev` were copied into the clone.
4. Tests: **264 passed** in the clone (same as the project).
5. Smoke path: `python scripts/run_pipeline.py --only 01_02 01_03 01_04 04_01 04_02 --skip-tests` -> PIPELINE OK, 5 steps, 114 s.
   `--dry-run` of the whole pipeline also works.

## Regenerated outputs compared with the project
| Output | Result |
|---|---|
| `news.parquet`, `behaviors.parquet`, `impressions.parquet` | identical frames (65,238 / 230,117 / 8,584,442 rows) |
| `data_summary.json` | identical except `run_info` (package versions, git commit, timestamp) |
| `archive/figures/EDA/eda_stats.json` | identical |
| M0 popularity: frozen setting + hash, valid per-impression metrics, summary row | identical (hash 5ade23b12a, max difference 0.0) |
| M1 TF-IDF: frozen setting + hash, valid per-impression metrics, summary row | identical (hash 4c4afdcfbc, max difference 0.0) |

Not rerun in the clone: GPU steps (GoEmotions labelling, SBERT encoding, NRMS) and the multi-hour tree tuning; those are covered by their
saved run files, the frozen settings with hashes, and the earlier rebuild proof (`docs/reports/tidy_report.md`).
