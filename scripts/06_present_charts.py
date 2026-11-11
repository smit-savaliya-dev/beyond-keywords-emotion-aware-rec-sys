"""Presentation charts (archive/figures/present/), built only from saved result tables: no training, no dev scoring, no change to
any existing result file or chart.

Reads:  results/tables/{model_ladder, m2m3_comparison, behaviour_e7_final_family, v7_dev_families, v7_subgroups, v7_survival_dev,
        v7_single_feature_auc, v7_fold_evidence, v7_fold_means, behaviour_e5_within_category_auc,
        behaviour_e5_m3_gain_within_category, emotion_label_variants_grid}.csv, emotion_weighted_vs_combined.json,
        emotion_v2_choice.json; archive/figures/EDA/eda_stats.json (part sizes and dates); src.config (history length).
Writes: archive/figures/present/NN_*.png (16:9, 200 dpi) + one CSV per chart with the plotted values and the exact label text,
        00_overview.png, README.md. A chart whose source file is missing is skipped and named in the output.
Every drawn number is read back from the chart's CSV after it is written, so labels and CSVs cannot disagree.
Run:    python scripts/06_present_charts.py [--out DIR]
"""
import argparse
import json
import textwrap
import time

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from src.config import CONGRUENCE_MAX_HISTORY, ROOT, TABLES

OUT = ROOT / "archive" / "figures" / "present"
EDA_STATS = ROOT / "archive" / "figures" / "EDA" / "eda_stats.json"
# Okabe-Ito (colour-blind safe). Blue = without emotion, orange = with emotion, grey = not significant; vermilion = significant loss.
BLUE, ORANGE, GREY, VERMILION, GREEN, PURPLE, YELLOW = "#0072B2", "#E69F00", "#999999", "#D55E00", "#009E73", "#CC79A7", "#F0E442"
INK, MUTED = "#1a1a1a", "#555555"
R1_WORDS, R2_WORDS = "previous days only (R1)", "with same-day clicks (R2)"
ALPHA = 0.05
plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14,
                     "axes.edgecolor": GREY, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "font.family": "DejaVu Sans"})


class Missing(Exception):
    """A source file or row the chart needs is not in the saved results."""


def src(name):
    p = TABLES / name
    if not p.exists():
        raise Missing(f"results/tables/{name}")
    return p


def table(name):
    return pd.read_csv(src(name))


def jfile(path, label):
    if not path.exists():
        raise Missing(label)
    return json.loads(path.read_text())


def one(t, name, **eq):
    """Exactly one row of `t` where every column equals the given value."""
    m = np.ones(len(t), bool)
    for k, v in eq.items():
        m &= (t[k] == v).to_numpy()
    if m.sum() != 1:
        raise Missing(f"results/tables/{name}: {m.sum()} rows for {eq}")
    return t[m].iloc[0]


def gain_txt(x):
    return f"{x:+.4f}"


def auc_txt(x):
    return f"{x:.3f}"


def holm_txt(p):
    return "Holm p < 0.001" if p < 0.001 else f"Holm p = {p:.2f}" if p >= 0.01 else f"Holm p = {p:.3f}"


# ---------------------------------------------------------------- figure helpers
def figure(title, caption):
    """16:9 white figure, plain-language title at the top, one-line caption at the bottom."""
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.04, 0.955, title, fontsize=25, fontweight="bold", va="top", ha="left")
    fig.text(0.04, 0.035, textwrap.fill(caption, 128), fontsize=15, color=MUTED, va="bottom", ha="left")
    return fig


def tidy(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color="#e6e6e6", lw=1)


def save(fig, out, name):
    for attempt in range(5):           # an open image preview can lock the file for a moment on Windows
        try:
            fig.savefig(out / name, dpi=200, facecolor="white")
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2)
    plt.close(fig)


def write_csv(df, out, stem):
    """Write the chart data and read it back: the chart is drawn from the read-back frame."""
    p = out / f"{stem}.csv"
    df.to_csv(p, index=False)
    text = {c: str for c in df.columns if c.endswith("label") or c in ("name", "text", "setting", "series")}
    return pd.read_csv(p, keep_default_na=False, na_values=[""], dtype=text)


def box(ax, x, y, w, h, text, face, edge, size=15, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=face, ec=edge, lw=2.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, weight=weight, linespacing=1.35)


def arrow(ax, a, b, color=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=26, lw=2.4, color=color, shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- source rows
def twin_gains():
    """The five emotion-vs-twin comparisons on the final test day (dev, saved predictions)."""
    m2m3 = table("m2m3_comparison.csv")
    a = one(m2m3, "m2m3_comparison.csv", model_a="M2_sbert", model_b="M3_sbert_emotion", metric="auc", part="dev")
    e7 = table("behaviour_e7_final_family.csv")
    b = one(e7, "behaviour_e7_final_family.csv", comparison="M3b+ (emotion, 50 items, no decay) - M2 Sentence-BERT", metric="auc", part="dev")
    fam = table("v7_dev_families.csv")
    r1 = one(fam, "v7_dev_families.csv", comparison="M5v6_R1 - M4v6_R1 (trees, R1)", family="R1 primary (emotion vs twin)")
    r2 = one(fam, "v7_dev_families.csv", comparison="M5v6 - M4v6 (trees)", family="R2 primary (plan v6, unchanged)")
    m7 = one(fam, "v7_dev_families.csv", comparison="M7 - M6 (NRMS)", family="R1 primary (emotion vs twin)")
    rows = [("M3 vs M2", "content model (Sentence-BERT)", a, "m2m3_comparison.csv"),
            ("M3b+ vs M2", "content model, week-7 profile", b, "behaviour_e7_final_family.csv"),
            ("M5v6 vs M4v6", f"feature ranker, {R1_WORDS}", r1, "v7_dev_families.csv"),
            ("M5v6 vs M4v6", f"feature ranker, {R2_WORDS}", r2, "v7_dev_families.csv"),
            ("M7 vs M6", "neural reader model (NRMS)", m7, "v7_dev_families.csv")]
    return pd.DataFrame([{"comparison": c, "setting": s, "gain": r["mean_diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                          "p_holm": r["p_holm"], "significant": bool(r["p_holm"] < ALPHA), "impressions": int(r["n"]), "source": f}
                         for c, s, r, f in rows])


def dev_sub(split):
    t = table("v7_subgroups.csv")
    return t[(t["part"] == "dev") & (t["regime"] == "R1") & (t["split"] == split)].copy()


# ---------------------------------------------------------------- charts
def c01_pipeline(out):
    steps = ["MIND-small\nclick logs", "Article emotion\nscores\n(GoEmotions + NRC,\ntitle + abstract)",
             f"Reader emotional\nprofile\n(last {CONGRUENCE_MAX_HISTORY} clicks)", "Match score\n(reader vs\narticle)",
             "Content score\n+ emotion score", "Final\nranking"]
    d = write_csv(pd.DataFrame({"step": range(1, 7), "text": [s.replace("\n", " ") for s in steps],
                                "source": ["method"] * 5 + ["method"]}).assign(
        history_items=CONGRUENCE_MAX_HISTORY, history_source="src/config.py CONGRUENCE_MAX_HISTORY"), out, "01_pipeline")
    fig = figure("How the emotion-aware ranker works",
                 "Each article gets emotion scores; each reader gets a profile from recent clicks; their match is added to the content score.")
    ax = fig.add_axes([0.02, 0.14, 0.96, 0.7])
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.axis("off")
    face = ["#eef4fa", "#fdf1dc", "#fdf1dc", "#fdf1dc", "#eef4fa", "#e8f5ef"]
    edge = [BLUE, ORANGE, ORANGE, ORANGE, BLUE, GREEN]
    for i, s in enumerate(steps):
        box(ax, i + 0.5, 0.5, 0.84, 0.42, s, face[i], edge[i], size=15)
        if i < 5:
            arrow(ax, (i + 0.93, 0.5), (i + 1.07, 0.5))
    ax.legend(handles=[Patch(fc="#fdf1dc", ec=ORANGE, lw=2, label="emotion part"), Patch(fc="#eef4fa", ec=BLUE, lw=2, label="data and content part")],
              loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=15)
    assert len(d) == 6
    save(fig, out, "01_pipeline.png")
    return d


def c02_time_split(out):
    st = jfile(EDA_STATS, "archive/figures/EDA/eda_stats.json")["parts"]
    rows = []
    for part, name in (("fit", "Train"), ("valid", "Validate"), ("dev", "Test")):
        p = st[part]
        a, b = pd.Timestamp(p["first_time"]).normalize(), pd.Timestamp(p["last_time"]).normalize()
        rows.append({"part": part, "name": name, "first_day": a.strftime("%Y-%m-%d"), "last_day": b.strftime("%Y-%m-%d"),
                     "days": (b - a).days + 1, "impressions": int(p["impressions"]), "label": f"{int(p['impressions']):,}" + chr(10) + "impressions"})
    d = write_csv(pd.DataFrame(rows).assign(source="archive/figures/EDA/eda_stats.json"), out, "02_time_split")
    fig = figure("We train on the past and test on a later day",
                 "Models learn from 9-13 Nov, settings are chosen on 14 Nov, and the final test is 15 Nov; no setting is chosen on the test day.")
    ax = fig.add_axes([0.05, 0.2, 0.9, 0.6])
    start = pd.Timestamp(d["first_day"].iloc[0])
    total = (pd.Timestamp(d["last_day"].iloc[-1]) - start).days + 1
    for (_, r), c in zip(d.iterrows(), [BLUE, ORANGE, GREEN]):
        x0 = (pd.Timestamp(r["first_day"]) - start).days
        ax.barh(0, r["days"], left=x0, height=0.5, color=c, ec="white", lw=3)
        a, b = pd.Timestamp(r["first_day"]), pd.Timestamp(r["last_day"])
        when = f"{a.day}-{b.day} Nov" if r["days"] > 1 else f"{a.day} Nov"
        ax.text(x0 + r["days"] / 2, 0, f"{r['name']}\n{when}", ha="center", va="center", fontsize=19, weight="bold", color="white")
        ax.text(x0 + r["days"] / 2, -0.42, r["label"], ha="center", va="top", fontsize=17, color=INK)
    for k in range(total + 1):
        ax.axvline(k, ymin=0.35, ymax=0.65, color="white", lw=1)
    ax.set_xticks([k + 0.5 for k in range(total)], [f"{(start + pd.Timedelta(days=k)).day} Nov" for k in range(total)])
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_xlim(0, total)
    ax.set_ylim(-0.9, 0.55)
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.xaxis.tick_top()
    save(fig, out, "02_time_split.png")
    return d


def c03_neutral(out):
    base = jfile(TABLES / "emotion_weighted_vs_combined.json", "results/tables/emotion_weighted_vs_combined.json")
    choice = jfile(TABLES / "emotion_v2_choice.json", "results/tables/emotion_v2_choice.json")
    grid = table("emotion_label_variants_grid.csv")
    t = float(choice["chosen_t"])
    after = grid[(grid["variant"] == "v2b") & np.isclose(grid["t"], t)]
    if len(after) != 1:
        raise Missing(f"results/tables/emotion_label_variants_grid.csv: no v2b row at t={t}")
    d = write_csv(pd.DataFrame([
        {"stage": "before: top label of the classifier", "neutral_pct": round(float(base["weighted"]["dominant_pct"]["neutral"]), 1),
         "source": "emotion_weighted_vs_combined.json weighted.dominant_pct.neutral"},
        {"stage": f"after: any emotion label above t = {t:g} (stance labels excluded)", "neutral_pct": round(float(after["pct_neutral"].iloc[0]), 1),
         "source": f"emotion_label_variants_grid.csv variant v2b, t = {t:g} (t from emotion_v2_choice.json)"}]).assign(
        label=lambda x: x["neutral_pct"].map(lambda v: f"{v:.0f}%")), out, "03_neutral_fix")
    fig = figure(f"After the labelling fix, far fewer articles count as neutral ({d['label'][0]} to {d['label'][1]})",
                 "Same classifier, new rule: an article is emotional if any emotion label scores above the threshold (stance labels excluded).")
    ax = fig.add_axes([0.1, 0.17, 0.85, 0.66])
    tidy(ax)
    x = np.arange(2)
    ax.bar(x, d["neutral_pct"], width=0.5, color=[GREY, GREEN])
    for i, r in d.iterrows():
        ax.text(i, r["neutral_pct"] + 1.5, r["label"], ha="center", va="bottom", fontsize=24, weight="bold")
    ax.set_xticks(x, ["Before the fix\n(top label only)", "After the fix\n(any emotion above threshold)"], fontsize=17)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Articles labelled neutral (%)")
    save(fig, out, "03_neutral_fix.png")
    return d


def c04_ladder(out):
    lad = table("model_ladder.csv").set_index("model")
    pairs = [("Content models", "M2_sbert", "M3_sbert_emotion"), ("Feature rankers (LightGBM)", "M4_lightgbm", "M5_lightgbm_emotion"),
             ("Neural reader models", "M6_nrms", "M7b_nrms_emotion_gated")]
    floor = ["M0_popularity", "M1_tfidf"]
    for m in [m for _, a, b in pairs for m in (a, b)] + floor:
        if m not in lad.index:
            raise Missing(f"results/tables/model_ladder.csv: {m}")
    rows = [{"row": fam, "without_emotion": lad.loc[a, "label"], "with_emotion": lad.loc[b, "label"]} for fam, a, b in pairs]
    rows.append({"row": "Floor (simple baselines)", "without_emotion": " / ".join(lad.loc[m, "label"] for m in floor), "with_emotion": ""})
    d = write_csv(pd.DataFrame(rows).assign(source="model_ladder.csv"), out, "04_model_ladder")
    fig = figure("Every emotion model has a twin without emotion",
                 "Each pair differs only by the emotion part, so the gain of the orange model over its blue twin is the effect of emotion.")
    ax = fig.add_axes([0.03, 0.13, 0.94, 0.73])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    ax.text(4.2, 4.1, "Without emotion", ha="center", fontsize=19, weight="bold", color=BLUE)
    ax.text(8.0, 4.1, "With emotion", ha="center", fontsize=19, weight="bold", color=VERMILION)
    for i, r in d.iterrows():
        y = 3.35 - i * 0.98
        ax.text(0.1, y, r["row"], ha="left", va="center", fontsize=17, weight="bold", color=INK)
        if isinstance(r["with_emotion"], str) and r["with_emotion"]:
            box(ax, 4.2, y, 2.9, 0.62, r["without_emotion"], "#eef4fa", BLUE, size=17)
            box(ax, 8.0, y, 2.9, 0.62, r["with_emotion"], "#fdf1dc", ORANGE, size=17)
            arrow(ax, (5.72, y), (6.48, y))
            ax.text(6.1, y + 0.36, "+ emotion", ha="center", fontsize=14, color=MUTED)
        else:
            box(ax, 4.2, y, 2.9, 0.62, r["without_emotion"], "#f2f2f2", GREY, size=17)
            ax.text(6.0, y, "reference points: every model should beat these", ha="left", va="center", fontsize=15, color=MUTED)
    save(fig, out, "04_model_ladder.png")
    return d


def c05_all_models(out):
    lad = table("model_ladder.csv")
    order = ["M0_popularity", "M1_tfidf", "M2_sbert", "M3_sbert_emotion", "M3b_plus", "M4_lightgbm", "M5_lightgbm_emotion",
             "M6_nrms", "M7b_nrms_emotion_gated"]
    short = {"M0_popularity": "M0\npopularity", "M1_tfidf": "M1\nTF-IDF", "M2_sbert": "M2\nSBERT", "M3_sbert_emotion": "M3\nM2+emo",
             "M3b_plus": "M3b+\nM2+emo", "M4_lightgbm": "M4\nLightGBM", "M5_lightgbm_emotion": "M5\nM4+emo", "M6_nrms": "M6\nNRMS",
             "M7b_nrms_emotion_gated": "M7\nM6+emo"}
    rows = [one(lad, "model_ladder.csv", model=m, part="dev") for m in order]
    d = write_csv(pd.DataFrame([{"model": r["model"], "name": short[r["model"]].replace("\n", " "), "emotion": bool(r["emotion"]),
                                 "test_auc": round(float(r["auc"]), 3), "label": auc_txt(r["auc"]), "source": "model_ladder.csv (dev)"}
                                for r in rows]).assign(random=0.5), out, "05_all_models_auc")
    fig = figure("Every model beats random; emotion changes AUC by a few thousandths",
                 "Final-test AUC, 15 Nov, previous-days features only. Blue = without emotion, orange = with emotion. Axis starts at 0.45.")
    ax = fig.add_axes([0.07, 0.2, 0.9, 0.63])
    tidy(ax)
    x = np.arange(len(d))
    ax.bar(x, d["test_auc"], width=0.66, color=[ORANGE if e else BLUE for e in d["emotion"]], bottom=0)
    for i, r in d.iterrows():
        ax.text(i, r["test_auc"] + 0.004, r["label"], ha="center", va="bottom", fontsize=15, weight="bold")
    ax.axhline(0.5, color=INK, ls="--", lw=1.8)
    ax.text(len(d) - 0.5, 0.503, "random (0.50)", ha="right", va="bottom", fontsize=15)
    ax.set_xticks(x, [short[m] for m in d["model"]], fontsize=15)
    ax.set_ylim(0.45, 0.70)
    ax.set_ylabel("AUC on the final test day")
    ax.legend(handles=[Patch(color=BLUE, label="without emotion"), Patch(color=ORANGE, label="with emotion")], loc="upper left", frameon=False)
    save(fig, out, "05_all_models_auc.png")
    return d


def gain_bars(ax, d, names, sizes=None):
    """Horizontal gain bars with 95% CIs, top to bottom; orange = significant gain, vermilion = significant loss, grey = not significant."""
    y = np.arange(len(d))[::-1]
    col = [GREY if not s else (ORANGE if g > 0 else VERMILION) for g, s in zip(d["gain"], d["significant"])]
    ax.barh(y, d["gain"], height=0.62, color=col)
    ax.errorbar(d["gain"], y, xerr=[d["gain"] - d["ci_low"], d["ci_high"] - d["gain"]], fmt="none", ecolor=INK, elinewidth=1.8, capsize=6)
    span = float(d["ci_high"].max() - min(d["ci_low"].min(), 0))
    for yy, (_, r) in zip(y, d.iterrows()):
        right = r["gain"] >= 0
        x = (r["ci_high"] if right else r["ci_low"]) + (0.012 if right else -0.012) * span
        ax.text(x, yy, r["label"], va="center", ha="left" if right else "right", fontsize=16, weight="bold")
    ax.set_yticks(y, names)
    ax.axvline(0, color=INK, lw=1.4)
    tidy(ax, grid="x")
    return y


def c06_gain_vs_twin(out):
    g = twin_gains()
    g["label"] = g["gain"].map(gain_txt)
    d = write_csv(g, out, "06_emotion_gain_vs_twin")
    names = [f"{r['comparison']}\n{r['setting']}" +("" if r["significant"] else "\nnot significant (" + holm_txt(r["p_holm"]) + ")")
             for _, r in d.iterrows()]
    fig = figure("Emotion helps a little next to its twin; clearest with previous-days features",
                 "Gain of each emotion model over its no-emotion twin, final test day, 95% CIs over impressions. Grey = Holm p > 0.05.")
    ax = fig.add_axes([0.37, 0.17, 0.58, 0.67])
    gain_bars(ax, d, names)
    ax.set_xlabel("AUC gain over the twin (final test day)")
    lo, hi = float(d["ci_low"].min()), float(d["ci_high"].max())
    ax.set_xlim(min(lo, 0) - 0.0008, hi + 0.0018)
    save(fig, out, "06_emotion_gain_vs_twin.png")
    return d


def c07_consistency(out):
    t = dev_sub("consistency tertile")
    order = [("low", "Most varied readers"), ("mid", "Middle"), ("high", "Most consistent readers")]
    rows = [one(t, "v7_subgroups.csv", group=g) for g, _ in order]
    d = write_csv(pd.DataFrame([{"group": r["group"], "name": n, "impressions": int(r["impressions"]), "gain": r["gain"], "ci_low": r["ci_low"],
                                 "ci_high": r["ci_high"], "p_holm": r["p_holm"], "significant": bool(r["p_holm"] < ALPHA),
                                 "label": gain_txt(r["gain"]), "regime": R1_WORDS, "source": "v7_subgroups.csv (dev, R1)"}
                                for r, (_, n) in zip(rows, order)]), out, "07_gain_by_consistency")
    fig = figure("Emotion helps all readers, most of all those with consistent emotional tastes",
                 f"Gain of M5v6 over its twin M4v6, {R1_WORDS}, final test day, by reader consistency. Bars: 95% CI.")
    ax = fig.add_axes([0.08, 0.2, 0.88, 0.63])
    tidy(ax)
    x = np.arange(len(d))
    ax.bar(x, d["gain"], width=0.55, color=[ORANGE if s else GREY for s in d["significant"]])
    ax.errorbar(x, d["gain"], yerr=[d["gain"] - d["ci_low"], d["ci_high"] - d["gain"]], fmt="none", ecolor=INK, elinewidth=1.8, capsize=8)
    for i, r in d.iterrows():
        ax.text(i, r["ci_high"] + 0.0002, r["label"], ha="center", va="bottom", fontsize=19, weight="bold")
    ax.set_xticks(x, [f"{r['name']}\n({r['impressions']:,} impressions)" for _, r in d.iterrows()], fontsize=16)
    ax.set_ylim(0, float(d["ci_high"].max()) * 1.2)
    ax.set_ylabel("AUC gain over the twin")
    save(fig, out, "07_gain_by_consistency.png")
    return d


def c08_category(out):
    t = dev_sub("clicked category").sort_values("gain", ascending=False)
    d = write_csv(pd.DataFrame({"category": t["group"], "impressions": t["impressions"].astype(int), "gain": t["gain"], "ci_low": t["ci_low"],
                                "ci_high": t["ci_high"], "p_holm": t["p_holm"], "significant": t["p_holm"] < ALPHA,
                                "label": t["gain"].map(gain_txt), "regime": R1_WORDS,
                                "source": "v7_subgroups.csv (dev, R1, split 'clicked category')"}), out, "08_gain_by_category")
    names = [f"{r['category']} ({r['impressions']:,})" for _, r in d.iterrows()]
    fig = figure("Gains are largest for TV, news and sports clicks; music clicks lose",
                 f"Gain over the twin, {R1_WORDS}, final test day, by the clicked article's category. Grey = Holm p > 0.05.")
    ax = fig.add_axes([0.2, 0.17, 0.75, 0.67])
    gain_bars(ax, d, names)
    ax.set_xlabel("AUC gain over the twin (final test day)")
    lo, hi = float(d["ci_low"].min()), float(d["ci_high"].max())
    ax.set_xlim(lo - 0.012, hi + 0.008)
    ax.legend(handles=[Patch(color=ORANGE, label="significant gain"), Patch(color=VERMILION, label="significant loss"),
                       Patch(color=GREY, label="not significant")], loc="lower right", frameon=False)
    save(fig, out, "08_gain_by_category.png")
    return d


def c09_not_topic(out):
    a = table("behaviour_e5_within_category_auc.csv")
    emo = one(a, "behaviour_e5_within_category_auc.csv", group="all", part="valid", score="personal emotion match alone")
    ctl = one(a, "behaviour_e5_within_category_auc.csv", group="all", part="valid", score="labels shuffled (control)")
    g = table("behaviour_e5_m3_gain_within_category.csv")
    gv = one(g, "behaviour_e5_m3_gain_within_category.csv", group="all", part="valid")
    gd = one(g, "behaviour_e5_m3_gain_within_category.csv", group="all", part="dev")
    rows = [{"panel": "A", "name": "Emotion match alone", "value": emo["mean"], "ci_low": emo["ci_low"], "ci_high": emo["ci_high"],
             "p": emo["p_vs_ref"], "n": int(emo["n"]), "label": auc_txt(emo["mean"]), "part": "valid",
             "source": "behaviour_e5_within_category_auc.csv (group all)"},
            {"panel": "A", "name": "Same, labels shuffled (control)", "value": ctl["mean"], "ci_low": ctl["ci_low"], "ci_high": ctl["ci_high"],
             "p": ctl["p_vs_ref"], "n": int(ctl["n"]), "label": auc_txt(ctl["mean"]), "part": "valid",
             "source": "behaviour_e5_within_category_auc.csv (group all)"},
            {"panel": "B", "name": "Validation day (14 Nov)", "value": gv["mean"], "ci_low": gv["ci_low"], "ci_high": gv["ci_high"],
             "p": gv["p_vs_ref"], "n": int(gv["n"]), "label": gain_txt(gv["mean"]), "part": "valid",
             "source": "behaviour_e5_m3_gain_within_category.csv (group all)"},
            {"panel": "B", "name": "Final test day (15 Nov)", "value": gd["mean"], "ci_low": gd["ci_low"], "ci_high": gd["ci_high"],
             "p": gd["p_vs_ref"], "n": int(gd["n"]), "label": gain_txt(gd["mean"]), "part": "dev",
             "source": "behaviour_e5_m3_gain_within_category.csv (group all)"}]
    d = write_csv(pd.DataFrame(rows).assign(significant=lambda x: (x["ci_low"] > np.where(x["panel"] == "A", 0.5, 0.0))), out, "09_not_topic")
    fig = figure("Emotion still ranks clicks among articles of the same category",
                 "Scores compared only among candidates of one category, pooled. A: AUC vs chance (0.50). B: M3 minus M2. Bars: 95% CI.")
    A, B = d[d["panel"] == "A"].reset_index(drop=True), d[d["panel"] == "B"].reset_index(drop=True)
    ax1, ax2 = fig.add_axes([0.07, 0.2, 0.39, 0.6]), fig.add_axes([0.59, 0.2, 0.38, 0.6])
    for ax, t, colors, ylim, base, ttl, ylab in (
            (ax1, A, [ORANGE, GREY], (0.45, 0.58), 0.5, "A. Emotion match alone, inside a category", "AUC (validation day)"),
            (ax2, B, [GREY if not s else ORANGE for s in B["significant"]], (-0.001, 0.006), 0.0, "B. M3 minus M2, inside a category",
             "AUC gain")):
        tidy(ax)
        x = np.arange(len(t))
        ax.bar(x, t["value"] - base, bottom=base, width=0.5, color=colors)
        ax.errorbar(x, t["value"], yerr=[t["value"] - t["ci_low"], t["ci_high"] - t["value"]], fmt="none", ecolor=INK, elinewidth=1.8, capsize=8)
        for i, r in t.iterrows():
            ax.text(i, r["ci_high"] + (ylim[1] - ylim[0]) * 0.02, r["label"], ha="center", va="bottom", fontsize=18, weight="bold")
        ax.axhline(base, color=INK, lw=1.6, ls="--" if base == 0.5 else "-")
        ax.set_xticks(x, [f"{r['name']}\n({r['n']:,} lists)" for _, r in t.iterrows()], fontsize=14)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(ttl, fontsize=17, loc="left", weight="bold", pad=12)
        ax.set_ylabel(ylab)
    ax1.text(1.45, 0.502, "chance", ha="right", va="bottom", fontsize=14)
    ax2.text(0, float(B["ci_high"].iloc[0]) + 0.0007, "not significant" if not B["significant"].iloc[0] else "", ha="center", fontsize=14, color=MUTED)
    save(fig, out, "09_not_topic.png")
    return d


def c10_same_day(out):
    lad = table("model_ladder.csv")
    m4 = one(lad, "model_ladder.csv", model="M4_lightgbm", part="dev")
    surv = table("v7_survival_dev.csv")
    if len(surv) != 1:
        raise Missing("results/tables/v7_survival_dev.csv")
    s = surv.iloc[0]
    sf = one(table("v7_single_feature_auc.csv"), "v7_single_feature_auc.csv", feature="ctr_24h")
    rows = [{"panel": "A", "name": "First M4\n(previous days)", "auc": s["dev_M4"], "source": "v7_survival_dev.csv dev_M4"},
            {"panel": "A", "name": f"M4v6\n{R2_WORDS}", "auc": s["dev_M4v6_R2"], "source": "v7_survival_dev.csv dev_M4v6_R2"},
            {"panel": "A", "name": f"M4v6\n{R1_WORDS}", "auc": s["dev_M4v6_R1"], "source": "v7_survival_dev.csv dev_M4v6_R1"},
            {"panel": "B", "name": "Click rate\nwith same-day clicks", "auc": sf["same_day_auc_R2"],
             "source": "v7_single_feature_auc.csv ctr_24h same_day_auc_R2 (valid day)"},
            {"panel": "B", "name": "Click rate\nprevious days only", "auc": sf["previous_days_auc_R1"],
             "source": "v7_single_feature_auc.csv ctr_24h previous_days_auc_R1 (valid day)"}]
    if not np.isclose(m4["auc"], s["dev_M4"]):
        raise Missing("model_ladder.csv and v7_survival_dev.csv disagree on M4")
    d = write_csv(pd.DataFrame(rows).assign(label=lambda x: x["auc"].map(auc_txt), name=lambda x: x["name"].str.replace("\n", " | ")),
                  out, "10_same_day_check")
    fig = figure("M4v6's big jump comes from same-day clicks",
                 "A: final-test AUC of the no-emotion ranker. B: the item's 24-hour click rate alone, validation day. Dashed = random.")
    ax1, ax2 = fig.add_axes([0.07, 0.2, 0.52, 0.6]), fig.add_axes([0.68, 0.2, 0.29, 0.6])
    for ax, t, ttl, ylab in ((ax1, d[d["panel"] == "A"], "A. Feature ranker without emotion (test day)", "AUC on the final test day"),
                             (ax2, d[d["panel"] == "B"], "B. Click rate alone (validation day)", "AUC")):
        t = t.reset_index(drop=True)
        tidy(ax)
        x = np.arange(len(t))
        cols = [BLUE if "same-day" in n else "#56B4E9" for n in t["name"]]
        ax.bar(x, t["auc"], width=0.55, color=cols)
        for i, r in t.iterrows():
            ax.text(i, r["auc"] + 0.004, r["label"], ha="center", va="bottom", fontsize=18, weight="bold")
        ax.axhline(0.5, color=INK, ls="--", lw=1.6)
        ax.set_xticks(x, [n.replace(" | ", "\n") for n in t["name"]], fontsize=14)
        ax.set_ylim(0.45, 0.75)
        ax.set_title(ttl, fontsize=17, loc="left", weight="bold", pad=12)
        ax.set_ylabel(ylab)
    ax1.legend(handles=[Patch(color=BLUE, label="uses same-day clicks"), Patch(color="#56B4E9", label="previous days only")],
               loc="upper right", frameon=False)
    save(fig, out, "10_same_day_check.png")
    return d


def c11_two_settings(out):
    fam = table("v7_dev_families.csv")
    r1 = one(fam, "v7_dev_families.csv", comparison="M5v6_R1 - M4v6_R1 (trees, R1)", family="R1 primary (emotion vs twin)")
    r2 = one(fam, "v7_dev_families.csv", comparison="M5v6 - M4v6 (trees)", family="R2 primary (plan v6, unchanged)")
    rows = []
    for setting, r in ((R1_WORDS, r1), (R2_WORDS, r2)):
        for kind, col, model in (("without emotion", "mean_a", r["comparison"].split(" - ")[1].split(" ")[0]),
                                 ("with emotion", "mean_b", r["comparison"].split(" - ")[0])):
            rows.append({"setting": setting, "series": kind, "model": model, "test_auc": r[col], "label": f"{r[col]:.4f}",
                         "pair_gain": r["mean_diff"], "gain_label": gain_txt(r["mean_diff"]), "p_holm": r["p_holm"],
                         "significant": bool(r["p_holm"] < ALPHA), "source": "v7_dev_families.csv"})
    d = write_csv(pd.DataFrame(rows), out, "11_two_settings")
    fig = figure("Emotion adds more when same-day clicks are not available",
                 "Final-test AUC. With same-day clicks, recent clicks already carry most of the signal, leaving little for emotion.")
    ax = fig.add_axes([0.08, 0.2, 0.88, 0.6])
    tidy(ax)
    w = 0.36
    for k, setting in enumerate([R1_WORDS, R2_WORDS]):
        t = d[d["setting"] == setting].reset_index(drop=True)
        for j, (_, r) in enumerate(t.iterrows()):
            x = k + (j - 0.5) * w
            ax.bar(x, r["test_auc"], width=w * 0.95, color=ORANGE if r["series"] == "with emotion" else BLUE)
            ax.text(x, r["test_auc"] + 0.002, r["label"], ha="center", va="bottom", fontsize=17, weight="bold")
        top = float(t["test_auc"].max())
        sig = bool(t["significant"].iloc[0])
        note = f"gain {t['gain_label'].iloc[0]}\n" + (holm_txt(t["p_holm"].iloc[0]) if sig else f"not significant ({holm_txt(t['p_holm'].iloc[0])})")
        ax.text(k, top + 0.011, note, ha="center", va="bottom", fontsize=17, color=INK if sig else MUTED, weight="bold" if sig else "normal")
    ax.set_xticks([0, 1], ["Previous days only (R1)\nstandard setting", "With same-day clicks (R2)\nreal-time setting"], fontsize=17)
    ax.set_ylim(0.6, 0.75)
    ax.set_ylabel("AUC on the final test day")
    ax.legend(handles=[Patch(color=BLUE, label="without emotion (M4v6)"), Patch(color=ORANGE, label="with emotion (M5v6)")],
              loc="upper left", frameon=False)
    save(fig, out, "11_two_settings.png")
    return d


def c12_folds(out):
    fe = table("v7_fold_evidence.csv")
    fm = table("v7_fold_means.csv").set_index("model")
    rows = []
    for reg, words, twin in (("R1", R1_WORDS, "M4v6_R1"), ("R2", R2_WORDS, "M4v6")):
        t = fe[fe["comparison"] == reg]
        if len(t) != 4 or twin not in fm.index:
            raise Missing(f"results/tables/v7_fold_evidence.csv / v7_fold_means.csv: {reg}")
        for _, r in t.iterrows():
            rows.append({"regime": reg, "setting": words, "fold": r["fold"], "emotion_model": r["model_b"], "twin": r["model_a"],
                         "gain": r["gain_seed_mean"], "label": gain_txt(r["gain_seed_mean"]), "seed_noise_sd": fm.loc[twin, "seed_sd"],
                         "source": "v7_fold_evidence.csv gain_seed_mean; v7_fold_means.csv seed_sd of the twin"})
    d = write_csv(pd.DataFrame(rows), out, "12_fold_evidence")
    fig = figure("With previous-days features the emotion gain repeats on every fold",
                 "Gain over the twin on four rolling test days (seed means). Shaded: +/-1 seed-noise SD of the twin. F2-F4 decide the twin rule.")
    ax = fig.add_axes([0.08, 0.17, 0.88, 0.66])
    tidy(ax)
    x = np.arange(4)
    for reg, col, ls, mk, off in (("R1", ORANGE, "-", "o", 1), ("R2", GREY, "--", "s", -1)):
        t = d[d["regime"] == reg].reset_index(drop=True)
        sd = float(t["seed_noise_sd"].iloc[0])
        ax.axhspan(-sd, sd, color=col, alpha=0.18 if reg == "R1" else 0.28, lw=0)
        ax.plot(x, t["gain"], ls=ls, marker=mk, ms=12, lw=3, color=col if reg == "R1" else MUTED)
        for i, r in t.iterrows():
            up = off > 0 or i == 3                     # last R2 point: label above-right, clear of the line and the zero axis
            dx = 0.08 if up else -0.08
            ax.text(i + dx, r["gain"] + (0.0004 if up else -0.0004), r["label"], ha="left" if dx > 0 else "right", va="bottom" if up else "top",
                    fontsize=15, weight="bold",
                    color=INK if reg == "R1" else MUTED)
    ax.axhline(0, color=INK, lw=1.3)
    ax.set_xticks(x, [f"{f}\n(test day {day} Nov)" for f, day in zip(d["fold"].unique(), (11, 12, 13, 14))], fontsize=15)
    ax.set_xlim(-0.3, 3.6)
    ax.set_ylabel("AUC gain over the twin")
    r1sd, r2sd = d.loc[d["regime"] == "R1", "seed_noise_sd"].iloc[0], d.loc[d["regime"] == "R2", "seed_noise_sd"].iloc[0]
    ax.legend(handles=[Line2D([], [], color=ORANGE, marker="o", lw=3, ms=11, label=f"{R1_WORDS}"),
                       Line2D([], [], color=MUTED, marker="s", ls="--", lw=3, ms=11, label=f"{R2_WORDS}"),
                       Patch(color=ORANGE, alpha=0.3, label=f"seed noise R1 (SD {r1sd:.4f})"),
                       Patch(color=GREY, alpha=0.45, label=f"seed noise R2 (SD {r2sd:.4f})")], loc="upper right", frameon=False)
    save(fig, out, "12_fold_evidence.png")
    return d


def c00_overview(out):
    panels = ["06_emotion_gain_vs_twin.png", "07_gain_by_consistency.png", "08_gain_by_category.png", "10_same_day_check.png",
              "11_two_settings.png", "05_all_models_auc.png"]
    missing = [p for p in panels if not (out / p).exists()]
    if missing:
        raise Missing("charts not built: " + ", ".join(missing))
    d = write_csv(pd.DataFrame({"position": range(1, 7), "chart": panels}), out, "00_overview")
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.5, 0.985, "Emotion-aware news ranking: main results", ha="center", va="top", fontsize=24, weight="bold")
    for i, p in enumerate(d["chart"]):
        ax = fig.add_axes([0.005 + (i % 3) * 0.332, 0.49 - (i // 3) * 0.475, 0.325, 0.44])
        ax.imshow(plt.imread(out / p))
        ax.axis("off")
    save(fig, out, "00_overview.png")
    return d


CHARTS = [("01_pipeline.png", "How it works", c01_pipeline), ("02_time_split.png", "How it works", c02_time_split),
          ("03_neutral_fix.png", "Problem", c03_neutral), ("04_model_ladder.png", "Models", c04_ladder),
          ("05_all_models_auc.png", "Result 1", c05_all_models), ("06_emotion_gain_vs_twin.png", "Result 1", c06_gain_vs_twin),
          ("07_gain_by_consistency.png", "Result 2", c07_consistency), ("08_gain_by_category.png", "Result 2", c08_category),
          ("09_not_topic.png", "Result 2", c09_not_topic), ("10_same_day_check.png", "Same-day check", c10_same_day),
          ("11_two_settings.png", "Same-day check", c11_two_settings), ("12_fold_evidence.png", "Conclusion", c12_folds),
          ("00_overview.png", "Conclusion", c00_overview)]


# ---------------------------------------------------------------- checks and README
def check_labels(name, d):
    """Every label string equals the formatted value in the same CSV row."""
    pairs = {"06": ("gain", gain_txt), "07": ("gain", gain_txt), "08": ("gain", gain_txt), "12": ("gain", gain_txt),
             "05": ("test_auc", auc_txt), "10": ("auc", auc_txt), "11": ("test_auc", lambda v: f"{v:.4f}")}
    k = name[:2]
    if k in pairs:
        col, f = pairs[k]
        bad = d[d[col].map(f) != d["label"]]
        assert bad.empty, f"{name}: labels differ from CSV values: {bad}"
    if k == "09":
        assert all(d["label"] == [auc_txt(v) if p == "A" else gain_txt(v) for v, p in zip(d["value"], d["panel"])])


def takeaways(data):
    """One plain sentence per chart, with numbers taken from the chart data."""
    g = data.get("06_emotion_gain_vs_twin.png")
    t = {}
    if "01_pipeline.png" in data:
        t["01_pipeline.png"] = "Articles get emotion scores, readers get an emotion profile, and their match is added to the content score."
    if (d := data.get("02_time_split.png")) is not None:
        t["02_time_split.png"] = (f"Train on {d['impressions'][0]:,} impressions (9-13 Nov), tune on {d['impressions'][1]:,} (14 Nov), "
                                  f"test once on {d['impressions'][2]:,} (15 Nov).")
    if (d := data.get("03_neutral_fix.png")) is not None:
        t["03_neutral_fix.png"] = f"The old rule called {d['label'][0]} of articles neutral; the fixed rule calls {d['label'][1]} neutral."
    if "04_model_ladder.png" in data:
        t["04_model_ladder.png"] = "Each emotion model is compared with a twin that is identical except for emotion."
    if (d := data.get("05_all_models_auc.png")) is not None:
        best = d.loc[d["test_auc"].idxmax()]
        t["05_all_models_auc.png"] = f"All models beat random (0.50); the best standard model is {best['name']} at {best['label']}."
    if g is not None:
        r1 = g[g["setting"].str.contains("R1")].iloc[0]
        t["06_emotion_gain_vs_twin.png"] = (f"Emotion beats its twin by small amounts; the clearest gain is {r1['label']} with previous-days "
                                            f"features; the same-day and NRMS gains are not significant.")
    if (d := data.get("07_gain_by_consistency.png")) is not None:
        t["07_gain_by_consistency.png"] = (f"The gain is positive for all readers and largest for the most consistent ones "
                                           f"({d['label'].iloc[-1]} vs {d['label'].iloc[0]}).")
    if (d := data.get("08_gain_by_category.png")) is not None:
        t["08_gain_by_category.png"] = (f"Biggest gain when the click is {d['category'].iloc[0]} ({d['label'].iloc[0]}); "
                                        f"music clicks lose ({d.set_index('category').loc['music', 'label']}).")
    if (d := data.get("09_not_topic.png")) is not None:
        t["09_not_topic.png"] = (f"Inside a single category the emotion match still beats chance (AUC {d['label'][0]} vs control "
                                 f"{d['label'][1]}), so emotion is not only topic.")
    if (d := data.get("10_same_day_check.png")) is not None:
        t["10_same_day_check.png"] = (f"With previous days only, M4v6 drops from {d['label'][1]} to {d['label'][2]}, below the first M4 "
                                      f"({d['label'][0]}); the click rate alone falls from {d['label'][3]} to {d['label'][4]}.")
    if (d := data.get("11_two_settings.png")) is not None:
        t["11_two_settings.png"] = (f"Emotion adds {d['gain_label'][0]} with previous days only and {d['gain_label'][2]} "
                                    "(not significant) with same-day clicks.")
    if (d := data.get("12_fold_evidence.png")) is not None:
        r1 = d[d["regime"] == "R1"]
        t["12_fold_evidence.png"] = (f"With previous-days features the gain is above zero on all four folds ({', '.join(r1['label'])}); "
                                     "with same-day clicks it stays near the noise.")
    if "00_overview.png" in data:
        t["00_overview.png"] = "One-page summary of charts 05, 06, 07, 08, 10 and 11."
    return t


SOURCES = {"01_pipeline.png": "method (src/config.py for the history length)", "02_time_split.png": "archive/figures/EDA/eda_stats.json",
           "03_neutral_fix.png": "emotion_weighted_vs_combined.json, emotion_label_variants_grid.csv, emotion_v2_choice.json",
           "04_model_ladder.png": "model_ladder.csv", "05_all_models_auc.png": "model_ladder.csv",
           "06_emotion_gain_vs_twin.png": "m2m3_comparison.csv, behaviour_e7_final_family.csv, v7_dev_families.csv",
           "07_gain_by_consistency.png": "v7_subgroups.csv", "08_gain_by_category.png": "v7_subgroups.csv",
           "09_not_topic.png": "behaviour_e5_within_category_auc.csv, behaviour_e5_m3_gain_within_category.csv",
           "10_same_day_check.png": "v7_survival_dev.csv, v7_single_feature_auc.csv, model_ladder.csv",
           "11_two_settings.png": "v7_dev_families.csv", "12_fold_evidence.png": "v7_fold_evidence.csv, v7_fold_means.csv",
           "00_overview.png": "the six chart PNGs"}


def readme(out, data, skipped):
    tk = takeaways(data)
    lines = ["# Presentation charts", "",
             "Built by `python scripts/06_present_charts.py` from saved result tables only (no training, no dev scoring). Each PNG has a CSV",
             "next to it with the plotted values and the exact label text. Source CSVs are in `results/tables/` unless a path is given.",
             "R1 = previous days only (standard setting); R2 = with same-day clicks (real-time setting). \"Final test\" = 15 Nov (dev).", "",
             "| Chart | Slide | Takeaway | Source |", "|---|---|---|---|"]
    for name, slide, _ in CHARTS:
        if name in data:
            lines.append(f"| `{name}` | {slide} | {tk.get(name, '')} | {SOURCES[name]} |")
    if skipped:
        lines += ["", "## Not built", ""] + [f"- `{n}`: missing {why}" for n, why in skipped]
    q = []
    if (g := data.get("06_emotion_gain_vs_twin.png")) is not None:
        for _, r in g.iterrows():
            q.append(f"- {r['comparison']} ({r['setting']}): {r['label']} AUC, 95% CI {r['ci_low']:+.4f} to {r['ci_high']:+.4f}, "
                     f"{holm_txt(r['p_holm'])}")
    if (d := data.get("07_gain_by_consistency.png")) is not None:
        q.append("- Gain by consistency (R1, test day): " + ", ".join(f"{n.lower()} {lab}" for n, lab in zip(d["name"], d["label"])))
    if (d := data.get("08_gain_by_category.png")) is not None:
        q.append("- Gain by clicked category (R1, test day): " + ", ".join(f"{c} {lab}" for c, lab in zip(d["category"], d["label"])))
    if (d := data.get("10_same_day_check.png")) is not None:
        q.append(f"- Feature ranker without emotion, test AUC: first M4 {d['label'][0]}, M4v6 with same-day clicks {d['label'][1]}, "
                 f"M4v6 previous days only {d['label'][2]}")
        q.append(f"- 24-hour click rate alone (validation day): with same-day clicks {d['label'][3]}, previous days only {d['label'][4]}")
    if (d := data.get("03_neutral_fix.png")) is not None:
        q.append(f"- Articles labelled neutral: {d['label'][0]} before the fix, {d['label'][1]} after")
    if (d := data.get("02_time_split.png")) is not None:
        q.append("- Impressions: " + ", ".join(f"{n.lower()} {v:,}" for n, v in zip(d["name"], d["impressions"])))
    lines += ["", "## Numbers to quote", ""] + q + [""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    data, skipped = {}, []
    for name, _, fn in CHARTS:
        try:
            d = fn(out)
        except Missing as e:
            skipped.append((name, str(e)))
            print(f"skipped {name}: missing {e}")
            continue
        check_labels(name, d)
        data[name] = d
        print("wrote", name)
    readme(out, data, skipped)
    return data, skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    main(type(OUT)(ap.parse_args().out))
