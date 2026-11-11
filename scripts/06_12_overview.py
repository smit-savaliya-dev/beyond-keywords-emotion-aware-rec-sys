"""One-page contact sheet of the six most useful week-7 charts, for the professor (archive/figures/v5_behaviour/overview.png).

Reads:  the PNGs written by scripts 06_01 to 06_11. Writes: archive/figures/v5_behaviour/overview.png.
Run:    python scripts/06_12_overview.py
"""
import _bootstrap  # noqa: F401
from src.behaviour_charts import contact_sheet
from src.config import BEHAVIOUR_FIGS

BEST = ["e7_final_comparison.png", "e1_gain_by_tertile.png", "e8_gain_by_persona.png", "e5_topic_vs_residual.png",
        "e7_m5b_variants_valid.png", "e9_diversity.png"]

if __name__ == "__main__":
    out = contact_sheet([BEHAVIOUR_FIGS / n for n in BEST], BEHAVIOUR_FIGS / "overview.png",
                        "Beyond Keywords, week 7: when does emotional context help ranking?", cols=3)
    print(out)
