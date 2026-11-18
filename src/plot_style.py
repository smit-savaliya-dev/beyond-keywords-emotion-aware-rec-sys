"""One naming and colour scheme for every final chart (scripts/07_final_figures.py).

Charts never show an internal code alone: DISPLAY gives the plain name. Colours are colour-blind safe (Okabe-Ito):
grey = baselines, blue = without emotion, orange = with emotion, light grey (+ the words "not significant") = Holm p > 0.05,
vermilion = a significant loss.
"""
import textwrap
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DISPLAY = {"M0_popularity": "Popularity", "M1_tfidf": "Keyword match (TF-IDF)", "M2_sbert": "Meaning match (Sentence-BERT)",
           "M3_sbert_emotion": "Meaning + emotion", "M3b_plus": "Meaning + emotion, tuned profile",
           "M4v6_R1": "LightGBM ranker", "M5v6_R1": "LightGBM + emotion", "M4v6": "LightGBM ranker", "M5v6": "LightGBM + emotion",
           "M6_nrms": "Neural ranker (NRMS)", "M7b_nrms_emotion_gated": "Neural + emotion (earlier version)", "M7d_nrms_emotion_fusion": "Neural + emotion",
           "B0_R1": "Combined ranker (LightGBM + neural)", "B1_R1": "Combined + emotion (own blend weight)", "B0": "Combined ranker (LightGBM + neural)",
           "B1": "Combined + emotion", "B1_R1_same_weight": "Combined + emotion", "B1_same_weight": "Combined + emotion"}
CODE = {"M0_popularity": "M0", "M1_tfidf": "M1", "M2_sbert": "M2", "M3_sbert_emotion": "M3", "M3b_plus": "M3b+", "M4v6_R1": "M4v6",
        "M5v6_R1": "M5v6", "M4v6": "M4v6", "M5v6": "M5v6", "M6_nrms": "M6", "M7b_nrms_emotion_gated": "M7b", "M7d_nrms_emotion_fusion": "M7d", "B0_R1": "B0",
        "B1_R1": "B1", "B0": "B0", "B1": "B1", "B1_R1_same_weight": "B1", "B1_same_weight": "B1"}
SETTING = {"Standard": "Standard setting (previous days only)", "Real-time": "Real-time setting (with same-day clicks)"}
SHORT_SETTING = {"Standard": "Standard (previous days only)", "Real-time": "Real-time (with same-day clicks)"}
FAMILIES = ["Baselines", "Meaning", "LightGBM", "Neural", "Combined"]

GREY, BLUE, ORANGE, LIGHT, VERMILION, GREEN, SKY = "#7f7f7f", "#0072B2", "#E69F00", "#d4d4d4", "#D55E00", "#009E73", "#56B4E9"
INK, MUTED = "#1a1a1a", "#555555"
NS = "not significant"
AUC_AXIS = "AUC: chance a clicked article ranks above a non-clicked one (0.50 = random)"
GAIN_AXIS = "AUC gain over the twin without emotion"
METRIC_AXIS = {"auc": "AUC", "mrr": "MRR: 1 / rank of the clicked article", "ndcg@5": "nDCG@5: quality of the top-5 order",
               "ndcg@10": "nDCG@10: quality of the top-10 order", "p@5": "Precision@5: share of the top 5 that is clicked",
               "r@10": "Recall@10: share of clicks found in the top 10", "hr@5": "Hit rate@5: a click in the top 5",
               "hr@10": "Hit rate@10: a click in the top 10"}

plt.rcParams.update({"font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
                     "axes.edgecolor": GREY, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "font.family": "DejaVu Sans", "svg.hashsalt": "fixed"})


def model_colour(emotion, family):
    return GREY if family == "Baselines" else (ORANGE if emotion else BLUE)


def gain_colour(gain, significant):
    return LIGHT if not significant else (ORANGE if gain > 0 else VERMILION)


def figure(title, caption, size=(16, 9)):
    """16:9 white figure: plain-language title on top, one-line "what this shows / what to take away" caption at the bottom."""
    fig = plt.figure(figsize=size, facecolor="white")
    fig.text(0.035, 0.96, title, fontsize=24, fontweight="bold", va="top", ha="left")
    fig.text(0.035, 0.03, textwrap.fill(caption, 135), fontsize=14.5, color=MUTED, va="bottom", ha="left")
    return fig


def tidy(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color="#e6e6e6", lw=1)


def save(fig, path):
    """200 dpi PNG with fixed metadata, so two runs give identical files."""
    for attempt in range(5):           # an open preview can lock the file for a moment on Windows
        try:
            fig.savefig(path, dpi=200, facecolor="white", metadata={"Software": None})
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2)
    plt.close(fig)


def write_csv(df, path):
    """Write the plotted numbers and read them back; the chart is drawn from the read-back frame so labels equal the CSV."""
    df.to_csv(path, index=False, lineterminator="\n")
    text = {c: str for c in df.columns if "label" in c or c in ("name", "text", "family", "setting", "metric_name", "group", "pair")}
    return pd.read_csv(path, keep_default_na=False, na_values=[""], dtype=text)


def gain_txt(x):
    return f"{x:+.4f}"


def auc_txt(x):
    return f"{x:.3f}"


def holm_txt(p):
    return "Holm p < 0.001" if p < 0.001 else (f"Holm p = {p:.3f}" if p < 0.01 else f"Holm p = {p:.2f}")


# ---------------------------------------------------------------------------------------------- week-9 scheme (scripts/08_week9_figures.py)
# Each approach has one colour family: light shade = without emotion, dark shade = with emotion (still readable in black and white).
# Dark shades checked with the dataviz palette validator (light surface): CVD worst pair 7.4 (purple/blue), normal-vision 19.4; every
# bar also carries its full name and value, so colour is never the only cue.
FAMILY_SHADES = {"Baselines": ("#a3a3a3", "#a3a3a3"), "Meaning": ("#dcb8d6", "#9e3d8f"), "LightGBM": ("#a9c9e8", "#1c6cb0"),
                 "Neural": ("#a6d4b0", "#1f6b35"), "Combined": ("#fcd3a1", "#dd7a0c")}
NS_GREY, LOSS_RED, GAIN_GREEN = "#c4c4c4", "#c0392b", "#2e8b57"
TABLE_RAMP = ("#f4f8f8", "#7fbcbc")          # one hue, light -> darker = better (table cells keep dark text)


def shade(family, emotion):
    return FAMILY_SHADES[family][1 if emotion else 0]
