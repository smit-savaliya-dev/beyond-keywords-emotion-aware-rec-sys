"""The final chart set (results/figures/), built only from saved tables: no training, no dev scoring.

Reads:  results/tables/final_all_metrics.csv, final_twin_gains_all_metrics.csv, final_dataset_overview.csv, summary.csv,
        v7_subgroups.csv, v7_single_feature_auc.csv, v7_fold_evidence.csv, v7_fold_means.csv, behaviour_e5_within_category_auc.csv,
        behaviour_e5_m3_gain_within_category.csv, emotion_weighted_vs_combined.json, emotion_label_variants_grid.csv,
        emotion_v2_choice.json, story_emotion_mix.csv, semeval2007_validation.csv, emotion_signal_v2.csv,
        v7_dev_shuffled_controls.csv; src.config (history length).
Writes: results/figures/{1_method,2_data,3_results,4_analysis}/*.png + one CSV per chart (plotted numbers and exact label text),
        results/figures/overview.png (+ CSV), results/figures/README.md. Names and colours: src/plot_style.py.
        results/figures/EDA/ is written by the EDA scripts (01_04, 02_05, 05_03), not by this script; the README lists it.
Run:    python scripts/07_final_figures.py [--out DIR]
"""
import argparse
import json

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from src import plot_style as ps
from src.config import CONGRUENCE_MAX_HISTORY, FIGURES, TABLES
from src.plot_style import (AUC_AXIS, BLUE, CODE, DISPLAY, GAIN_AXIS, GREEN, GREY, INK, LIGHT, MUTED, NS, ORANGE, SETTING, SKY,
                            VERMILION, auc_txt, gain_txt, holm_txt)

STD, RT = "Standard", "Real-time"
RANK8 = ["auc", "mrr", "ndcg@5", "ndcg@10", "p@5", "r@10", "hr@5", "hr@10"]
METRIC_LABEL = {"auc": "AUC", "mrr": "MRR", "ndcg@5": "nDCG@5", "ndcg@10": "nDCG@10", "p@5": "Precision@5", "r@10": "Recall@10",
                "hr@5": "Hit rate@5", "hr@10": "Hit rate@10"}
STD_ORDER = ["M0_popularity", "M1_tfidf", "M2_sbert", "M3_sbert_emotion", "M3b_plus", "M4v6_R1", "M5v6_R1", "M6_nrms",
             "M7b_nrms_emotion_gated", "B0_R1", "B1_R1"]
PAIR_SHORT = {"Meaning + emotion vs Meaning match": "Meaning + emotion",
              "Tuned profile vs Meaning match": "Meaning + emotion, tuned profile",
              "LightGBM + emotion vs LightGBM (Standard)": "LightGBM + emotion",
              "LightGBM + emotion vs LightGBM (Real-time)": "LightGBM + emotion (real-time)",
              "Neural + emotion vs Neural": "Neural + emotion",
              "Combined + emotion vs Combined (Standard, same weight)": "Combined + emotion",
              "Combined + emotion vs Combined (Real-time, same weight)": "Combined + emotion (real-time)"}
STD_PAIRS = ["Meaning + emotion vs Meaning match", "Tuned profile vs Meaning match", "LightGBM + emotion vs LightGBM (Standard)",
             "Neural + emotion vs Neural", "Combined + emotion vs Combined (Standard, same weight)"]


class Missing(Exception):
    pass


def tab(name):
    p = TABLES / name
    if not p.exists():
        raise Missing(f"results/tables/{name}")
    return pd.read_csv(p) if name.endswith(".csv") else json.loads(p.read_text(encoding="utf-8"))


def one(t, name, **eq):
    m = np.ones(len(t), bool)
    for k, v in eq.items():
        m &= (t[k] == v).to_numpy()
    if m.sum() != 1:
        raise Missing(f"results/tables/{name}: {int(m.sum())} rows for {eq}")
    return t[m].iloc[0]


def final_metrics():
    t = tab("final_all_metrics.csv")
    return t[~t["model"].str.endswith("_same_weight")]


def gains():
    return tab("final_twin_gains_all_metrics.csv")


def box(ax, x, y, w, h, text, face, edge, size=15, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=face, ec=edge, lw=2.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, weight=weight, linespacing=1.35)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=26, lw=2.4, color=MUTED))


def hbar_gains(ax, d, names, label_size=15):
    """Horizontal gain bars (top to bottom) with 95% CIs; colour by sign and Holm significance; 'not significant' in the name."""
    y = np.arange(len(d))[::-1]
    ax.barh(y, d["gain"], height=0.62, color=[ps.gain_colour(g, s) for g, s in zip(d["gain"], d["significant"])])
    ax.errorbar(d["gain"], y, xerr=[d["gain"] - d["ci_low"], d["ci_high"] - d["gain"]], fmt="none", ecolor=INK, elinewidth=1.6, capsize=5)
    span = float(max(d["ci_high"].max(), 0) - min(d["ci_low"].min(), 0))
    for yy, (_, r) in zip(y, d.iterrows()):
        right = r["gain"] >= 0
        x = (r["ci_high"] if right else r["ci_low"]) + (0.015 if right else -0.015) * span
        ax.text(x, yy, r["label"], va="center", ha="left" if right else "right", fontsize=label_size, weight="bold")
    ax.set_yticks(y, names)
    ax.axvline(0, color=INK, lw=1.3)
    ps.tidy(ax, grid="x")
    lo, hi = min(float(d["ci_low"].min()), 0), max(float(d["ci_high"].max()), 0)
    ax.set_xlim(lo - 0.22 * span, hi + 0.3 * span)


# ======================================================================= 1_method
def pipeline(out):
    steps = ["MIND click logs", "Article emotion\nscores\n(GoEmotions +\nNRC, title +\nabstract)",
             f"Reader profile\n(last {CONGRUENCE_MAX_HISTORY} clicks)", "Match score\n(reader vs article)", "Content score\n+ emotion score",
             "Ranking"]
    d = ps.write_csv(pd.DataFrame({"step": range(1, 7), "text": [s.replace("\n", " ") for s in steps],
                                   "history_items": CONGRUENCE_MAX_HISTORY}), out / "pipeline.csv")
    fig = ps.figure("How the emotion-aware ranker works",
                    "Articles get emotion scores, readers get a profile of the emotions they click, and their match is added to the content score.")
    ax = fig.add_axes([0.02, 0.14, 0.96, 0.7])
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.axis("off")
    face = ["#eef4fa", "#fdf1dc", "#fdf1dc", "#fdf1dc", "#eef4fa", "#e8f5ef"]
    edge = [BLUE, ORANGE, ORANGE, ORANGE, BLUE, GREEN]
    for i, s in enumerate(steps):
        box(ax, i + 0.5, 0.55, 0.85, 0.5, s, face[i], edge[i])
        if i < 5:
            arrow(ax, (i + 0.94, 0.55), (i + 1.06, 0.55))
    ax.legend(handles=[Patch(fc="#fdf1dc", ec=ORANGE, lw=2, label="emotion part"), Patch(fc="#eef4fa", ec=BLUE, lw=2, label="data and content part")],
              loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=15)
    ps.save(fig, out / "pipeline.png")
    return d


def time_split(out):
    t = tab("final_dataset_overview.csv")
    d = ps.write_csv(t.assign(days=[(pd.Timestamp(b) - pd.Timestamp(a)).days + 1 for a, b in zip(t["first_day"], t["last_day"])],
                              label=[f"{v:,} impressions" for v in t["impressions"]]), out / "time_split.csv")
    fig = ps.figure("We train on the past and test on a later day",
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
        ax.text(x0 + r["days"] / 2, -0.4, r["label"].replace(" impressions", "\nimpressions"), ha="center", va="top", fontsize=17)
    ax.set_xticks([k + 0.5 for k in range(total)], [f"{(start + pd.Timedelta(days=k)).day} Nov" for k in range(total)])
    ax.tick_params(axis="x", length=0, pad=10)
    ax.xaxis.tick_top()
    ax.set_xlim(0, total)
    ax.set_ylim(-0.95, 0.55)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ps.save(fig, out / "time_split.png")
    return d


def model_pairs(out):
    rows = [("Meaning models", "M2_sbert", "M3b_plus"), ("LightGBM rankers", "M4v6_R1", "M5v6_R1"),
            ("Neural rankers", "M6_nrms", "M7d_nrms_emotion_fusion"), ("Combined rankers", "B0_R1", "B1_R1_same_weight"),
            ("Baselines", "M0_popularity", "M1_tfidf")]                 # the final pairs (D109)
    name = {**DISPLAY, "M3b_plus": "Meaning + emotion", "B0_R1": "Combined ranker"}
    code = {**CODE, "B1_R1_same_weight": "B1 (same weight)"}
    d = ps.write_csv(pd.DataFrame([{"row": r, "left": name[a], "left_code": code[a], "right": name[b], "right_code": code[b]}
                                   for r, a, b in rows]), out / "model_pairs.csv")
    fig = ps.figure("Every emotion model has a twin without emotion",
                    "Each pair differs only by the emotion part, so the gain of the orange model over its blue twin is the effect of emotion.")
    ax = fig.add_axes([0.03, 0.12, 0.94, 0.75])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.text(4.3, 5.15, "Without emotion", ha="center", fontsize=19, weight="bold", color=BLUE)
    ax.text(8.15, 5.15, "With emotion", ha="center", fontsize=19, weight="bold", color=VERMILION)
    for i, r in d.iterrows():
        y = 4.4 - i * 0.98
        ax.text(0.05, y, r["row"], ha="left", va="center", fontsize=17, weight="bold")
        base = r["row"] == "Baselines"
        for x, name, code, face, edge in ((4.3, r["left"], r["left_code"], "#f2f2f2" if base else "#eef4fa", GREY if base else BLUE),
                                          (8.15, r["right"], r["right_code"], "#f2f2f2" if base else "#fdf1dc", GREY if base else ORANGE)):
            box(ax, x, y, 3.3, 0.66, name, face, edge, size=16)
            ax.text(x + 1.6, y - 0.27, code, ha="right", va="bottom", fontsize=12, color=MUTED)
        if not base:
            arrow(ax, (5.98, y), (6.47, y))
            ax.text(6.22, y + 0.37, "+ emotion", ha="center", va="bottom", fontsize=14, color=MUTED)
        else:
            ax.text(6.22, y + 0.42, "reference points, no emotion", ha="center", fontsize=14, color=MUTED)
    ps.save(fig, out / "model_pairs.png")
    return d


# ======================================================================= 2_data
def dataset_overview(out):
    t = tab("final_dataset_overview.csv")
    long = t.melt(id_vars=["name"], value_vars=["impressions", "users", "clicks"], var_name="measure", value_name="count")
    d = ps.write_csv(long.assign(label=long["count"].map(lambda v: f"{v:,}"), articles_total=int(t["articles_total"].iloc[0])),
                     out / "dataset_overview.csv")
    n_art = int(d["articles_total"].iloc[0])
    fig = ps.figure(f"MIND-small: {n_art:,} news articles and about {int(t['impressions'].sum()) // 1000}k impressions over seven days",
                    "Impressions = lists shown to a user; clicks = articles clicked in them. Each user is counted once per split.")
    ax = fig.add_axes([0.08, 0.17, 0.89, 0.66])
    ps.tidy(ax)
    measures, cols = ["impressions", "users", "clicks"], [BLUE, SKY, ORANGE]
    w = 0.26
    names = list(t["name"])
    for j, (m, c) in enumerate(zip(measures, cols)):
        sub = d[d["measure"] == m].set_index("name").loc[names]
        x = np.arange(len(names)) + (j - 1) * w
        ax.bar(x, sub["count"], width=w * 0.95, color=c, label=m)
        for xx, (_, r) in zip(x, sub.iterrows()):
            ax.text(xx, r["count"] * 1.01 + 2000, r["label"], ha="center", va="bottom", fontsize=13.5, weight="bold")
    days = [f"{a}\n({pd.Timestamp(f).day}-{pd.Timestamp(l).day} Nov)" if f != l else f"{a}\n({pd.Timestamp(f).day} Nov)"
            for a, f, l in zip(t["name"], t["first_day"], t["last_day"])]
    ax.set_xticks(np.arange(len(names)), days, fontsize=16)
    ax.set_ylabel("Count")
    ax.set_ylim(0, float(d["count"].max()) * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.legend(loc="upper right", frameon=False)
    ps.save(fig, out / "dataset_overview.png")
    return d


EMO_COLOURS = [("neutral", LIGHT), ("joy", ORANGE), ("sadness", BLUE), ("surprise", SKY), ("anger", VERMILION), ("fear", "#CC79A7")]


def neutral_fix(out):
    """Full emotion mix before and after the labelling fix; the neutral shares are checked against the tables that chose the rule."""
    base = tab("emotion_weighted_vs_combined.json")
    t = float(tab("emotion_v2_choice.json")["chosen_t"])
    grid = tab("emotion_label_variants_grid.csv")
    after = grid[(grid["variant"] == "v2b") & np.isclose(grid["t"], t)]
    if len(after) != 1:
        raise Missing("results/tables/emotion_label_variants_grid.csv: v2b row at the chosen t")
    mix = tab("story_emotion_mix.csv").set_index("Unnamed: 0")
    stages = [("Before the fix\n(top label only)", "v1 (1/3 title, argmax)", float(base["weighted"]["dominant_pct"]["neutral"])),
              (f"After the fix\n(any emotion above {t:g})", "v2b (v2 without stance labels)", float(after["pct_neutral"].iloc[0]))]
    rows = []
    for stage, key, neutral_check in stages:
        if key not in mix.index:
            raise Missing(f"results/tables/story_emotion_mix.csv: row {key}")
        assert abs(float(mix.loc[key, "neutral"]) - neutral_check) < 0.05, f"neutral share differs for {key}"
        for emo, _ in EMO_COLOURS:
            v = float(mix.loc[key, emo])
            rows.append({"stage": stage.replace("\n", " | "), "emotion": emo, "pct": v, "label": f"{v:.0f}%",
                         "source": f"story_emotion_mix.csv '{key}'"})
    d = ps.write_csv(pd.DataFrame(rows), out / "neutral_fix.csv")
    neu = d[d["emotion"] == "neutral"]["label"].tolist()
    fig = ps.figure(f"After the labelling fix, far fewer articles count as neutral ({neu[0]} to {neu[1]})",
                    "Share of all articles by main emotion label. Same classifier, new rule: an article is emotional if any emotion "
                    "label scores above the threshold (stance labels excluded).")
    ax = fig.add_axes([0.1, 0.2, 0.66, 0.65])
    ps.tidy(ax)
    st = list(dict.fromkeys(d["stage"]))
    bottom = np.zeros(len(st))
    for emo, col in EMO_COLOURS:
        v = d[d["emotion"] == emo].set_index("stage").loc[st]
        ax.bar(range(len(st)), v["pct"], bottom=bottom, width=0.5, color=col, ec="white", lw=1.5, label=emo)
        for i, (_, r) in enumerate(v.iterrows()):
            if r["pct"] >= 3:
                ax.text(i, bottom[i] + r["pct"] / 2, f"{emo} {r['label']}", ha="center", va="center", fontsize=16, weight="bold",
                        color="white" if emo in ("sadness", "anger") else INK)
        bottom += v["pct"].to_numpy()
    ax.set_xticks(range(len(st)), [s.replace(" | ", "\n") for s in st], fontsize=17)
    ax.set_xlim(-0.6, len(st) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of all articles (%)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=16, reverse=True)
    ps.save(fig, out / "neutral_fix.png")
    return d


def emotion_validation(out):
    t = tab("semeval2007_validation.csv")
    gold = [("valence", "go_vad_valence", "vad_valence")] + [(e, "goemotions_group", "nrc_emolex")
                                                             for e in ("sadness", "anger", "fear", "joy", "surprise")]
    rows = []
    for e, go, nrc in gold:
        for src, name in ((go, "GoEmotions (our article scores)"), (nrc, "NRC word lists")):
            r = one(t, "semeval2007_validation.csv", gold=e, source=src, kind="correlation")
            rows.append({"emotion": e, "source": name, "source_code": src, "spearman": r["spearman"], "ci_low": r["spearman_low"],
                         "ci_high": r["spearman_high"], "n": int(r["n"]), "label": f"{r['spearman']:.2f}"})
    d = ps.write_csv(pd.DataFrame(rows), out / "emotion_validation.csv")
    fig = ps.figure("Our emotion scores agree with human ratings better than the NRC word lists",
                    f"Spearman correlation with SemEval-2007 human ratings of {d['n'].iloc[0]:,} news headlines, 95% CIs. NRC = EmoLex for the "
                    "five emotions, VAD lexicon for valence. 0 = no agreement.")
    ax = fig.add_axes([0.08, 0.21, 0.89, 0.62])
    ps.tidy(ax)
    w = 0.38
    emos = [g[0] for g in gold]
    for j, (src, c) in enumerate((("GoEmotions (our article scores)", ORANGE), ("NRC word lists", GREY))):
        sub = d[d["source"] == src].reset_index(drop=True)
        x = np.arange(len(emos)) + (j - 0.5) * w
        ax.bar(x, sub["spearman"], width=w * 0.95, color=c, label=src)
        ax.errorbar(x, sub["spearman"], yerr=[sub["spearman"] - sub["ci_low"], sub["ci_high"] - sub["spearman"]], fmt="none", ecolor=INK,
                    elinewidth=1.5, capsize=5)
        for xx, (_, r) in zip(x, sub.iterrows()):
            ax.text(xx, max(r["ci_high"], 0) + 0.015, r["label"], ha="center", va="bottom", fontsize=15, weight="bold")
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(np.arange(len(emos)), ["Valence\n(positive vs negative)"] + [e.capitalize() for e in emos[1:]], fontsize=16)
    ax.set_ylabel("Agreement with human ratings\n(Spearman correlation)")
    ax.set_ylim(min(0, float(d["ci_low"].min())) - 0.05, 0.85)
    ax.legend(loc="upper right", frameon=False)
    ps.save(fig, out / "emotion_validation.png")
    return d


def emotion_match_signal(out):
    """Before any model: is the article closer to the reader's emotion profile when it is clicked? (D-analysis, fit and valid only)."""
    t = tab("emotion_signal_v2.csv")
    order = [("cong_pct_cos", "Plain match"), ("cong_pct_cos_personal", "Personal match\n(minus the average reader)"),
             ("cong_pct_cos_userspecific", "Strict personal match\n(minus another reader)")]
    rows = []
    for part, pname in (("fit", "Training days (9-13 Nov)"), ("valid", "Validation day (14 Nov)")):
        for f, name in order:
            r = one(t, "emotion_signal_v2.csv", feature=f, part=part)
            rows.append({"part": pname, "feature": f, "name": name.replace("\n", " | "), "dz": r["dz"], "ci_low": r["dz_ci_low"],
                         "ci_high": r["dz_ci_high"], "impressions": int(r["n_groups"]), "significant": bool(r["significant_holm"]),
                         "label": f"{r['dz']:.2f}"})
    d = ps.write_csv(pd.DataFrame(rows), out / "emotion_match_signal.csv")
    fig = ps.figure("Readers click articles whose emotions match what they read before",
                    "Effect size (dz) of the emotion match of clicked minus non-clicked articles in the same list, 95% CIs. Measured before "
                    "any model was built; the test day is not used. 0 = no difference.")
    ax = fig.add_axes([0.09, 0.2, 0.88, 0.63])
    ps.tidy(ax)
    w = 0.36
    for j, (part, c) in enumerate(zip(dict.fromkeys(d["part"]), (BLUE, ORANGE))):
        s = d[d["part"] == part].reset_index(drop=True)
        x = np.arange(len(s)) + (j - 0.5) * w
        ax.bar(x, s["dz"], width=w * 0.95, color=c, label=f"{part}, {s['impressions'].iloc[0]:,} lists")
        ax.errorbar(x, s["dz"], yerr=[s["dz"] - s["ci_low"], s["ci_high"] - s["dz"]], fmt="none", ecolor=INK, elinewidth=1.6, capsize=6)
        for xx, (_, r) in zip(x, s.iterrows()):
            ax.text(xx, r["ci_high"] + 0.006, r["label"], ha="center", va="bottom", fontsize=17, weight="bold")
    ax.set_xticks(range(len(order)), [n for _, n in order], fontsize=16)
    ax.set_ylim(0, float(d["ci_high"].max()) * 1.3)
    ax.set_ylabel("Clicked minus non-clicked\n(effect size dz)")
    ax.legend(loc="upper right", frameon=False)
    ps.save(fig, out / "emotion_match_signal.png")
    return d


# ======================================================================= 3_results
def _all_models_bars(out, name, metric, value_col, title, caption, ylim, ylabel, random_line):
    """Final models of the standard setting as bars grouped by family (shared by the AUC and the nDCG@10 chart)."""
    m = final_metrics()
    m = m[m["setting"] == STD].set_index("model").loc[STD_ORDER].reset_index()
    early = one(tab("summary.csv"), "summary.csv", model="M4_lightgbm", part="dev")
    d = ps.write_csv(pd.DataFrame({"model": m["model"], "name": m["model"].map(DISPLAY), "code": m["model"].map(CODE), "family": m["family"],
                                   "emotion": m["emotion"], value_col: m[metric].round(3), "label": m[metric].map(auc_txt)}).assign(
        **{f"earlier_lightgbm_{metric.replace('@', '')}": round(float(early[metric]), 3)}), out / f"{name}.csv")
    fig = ps.figure(title(d), caption, size=(16, 9))
    ax = fig.add_axes([0.07, 0.31, 0.91, 0.54])
    ps.tidy(ax)
    gap, xs, x, prev = 0.6, [], 0.0, None
    for fam in d["family"]:
        if prev is not None and fam != prev:
            x += gap
        xs.append(x)
        x += 1
        prev = fam
    xs = np.array(xs)
    ax.bar(xs, d[value_col] - ylim[0], bottom=ylim[0], width=0.8, color=[ps.model_colour(e, f) for e, f in zip(d["emotion"], d["family"])])
    step = (ylim[1] - ylim[0]) / 83
    for xx, (_, r) in zip(xs, d.iterrows()):
        ax.text(xx, r[value_col] + step, r["label"], ha="center", va="bottom", fontsize=14, weight="bold")
    if random_line:
        ax.axhline(0.5, color=INK, ls="--", lw=1.6)
        ax.text(xs[-1] + 0.45, 0.502, "random", ha="right", va="bottom", fontsize=14)
    wrap = {"Meaning match (Sentence-BERT)": "Meaning\nmatch\n(S-BERT)", "Keyword match (TF-IDF)": "Keyword\nmatch\n(TF-IDF)",
            "Meaning + emotion, tuned profile": "Meaning +\nemotion,\ntuned", "Combined ranker (LightGBM + neural)": "Combined\nranker",
            "Neural ranker (NRMS)": "Neural\nranker\n(NRMS)", "Meaning + emotion": "Meaning +\nemotion", "LightGBM + emotion": "LightGBM +\nemotion",
            "Neural + emotion": "Neural +\nemotion", "Combined + emotion": "Combined +\nemotion", "LightGBM ranker": "LightGBM\nranker"}
    ax.set_xticks(xs, [wrap.get(n, n) for n in d["name"]], fontsize=13)
    for fam in d["family"].unique():
        xx = xs[(d["family"] == fam).to_numpy()]
        ax.annotate(fam, xy=(xx.mean(), 0), xycoords=("data", "axes fraction"), xytext=(0, -92), textcoords="offset points",
                    ha="center", va="top", fontsize=16, weight="bold", color=MUTED)
    ax.set_ylim(*ylim)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylabel(ylabel)
    fig.text(0.035, 0.085, f"An earlier LightGBM without emotion scored {d.iloc[0, -1]:.3f} in the standard setting; "
             "see the full results table.", fontsize=14.5, color=MUTED)
    ps.save(fig, out / f"{name}.png")
    return d


def all_models_auc(out):
    return _all_models_bars(out, "all_models_auc", "auc", "test_auc",
                            lambda d: "Every final model beats random; emotion adds a little within each family",
                            "Final-test AUC (15 Nov), standard setting (previous days only). Grey = baselines, blue = without emotion, "
                            "orange = with emotion.", (0.45, 0.70), "AUC (0.50 = random)", True)


def all_models_ndcg10(out):
    def title(d):
        best = d.loc[d["test_ndcg10"].idxmax()]
        return f"Top-10 ranking quality (nDCG@10): best is {best['name']} at {best['label']}"
    return _all_models_bars(out, "all_models_ndcg10", "ndcg@10", "test_ndcg10", title,
                            "Final-test nDCG@10 (15 Nov), standard setting; 0 to 1. Grey = baselines, blue = without emotion, orange = "
                            "with emotion.", (0.30, 0.44), "nDCG@10 (axis starts at 0.30)", False)


STD_FINAL, RT_FINAL = "Standard setting: final models", "Real-time setting: final models"
EARLIER = "Standard setting: earlier versions and controls"
EVERY_RUN = ([(STD_FINAL, c, DISPLAY[c], CODE[c], "final") for c in STD_ORDER]
             + [(EARLIER, "M4_lightgbm", "Earlier LightGBM ranker (week 4)", "M4", "earlier"),
                (EARLIER, "M5_lightgbm_emotion", "Earlier LightGBM + emotion (week 4)", "M5", "earlier"),
                (EARLIER, "M5_congruence_only", "Earlier LightGBM + emotion match only", "M5 match-only", "earlier"),
                (EARLIER, "M5b_lightgbm_emotion", "LightGBM + emotion, week-7 version", "M5b", "earlier"),
                (EARLIER, "M5_permuted_items", "Control: earlier LightGBM, emotions shuffled", "M5 permuted", "control"),
                (EARLIER, "M7ctrl_nrms_permuted", "Control: neural, emotions shuffled", "M7ctrl", "control")]
             + [(RT_FINAL, c, DISPLAY[c], CODE[c], "final") for c in ("M4v6", "M5v6", "B0", "B1")])
WITH_EMOTION = ("M3", "M5", "M7", "B1")


def all_models_auc_every_run(out):
    """Every model ever scored on the final test day (from presentation/1_model_scores), with its setting and status made explicit."""
    fm = tab("final_all_metrics.csv").set_index("model")
    s = tab("summary.csv")
    s = s[s["part"] == "dev"].set_index("model")
    rows = []
    for group, code, name, short, kind in EVERY_RUN:
        if code in fm.index:
            auc, src = float(fm.loc[code, "auc"]), "final_all_metrics.csv"
        elif code in s.index:
            auc, src = float(s.loc[code, "auc"]), "summary.csv (dev)"
        else:
            raise Missing(f"results/tables/summary.csv: dev row for {code}")
        rows.append({"group": group, "model": code, "name": name, "code": short, "kind": kind,
                     "emotion": code.startswith(WITH_EMOTION), "baseline": code in ("M0_popularity", "M1_tfidf"),
                     "test_auc": auc, "label": auc_txt(auc), "source": src})
    d = ps.write_csv(pd.DataFrame(rows), out / "all_models_auc_every_run.csv")
    fig = ps.figure("AUC of every model run on the final test day",
                    "Final test (15 Nov), bars start at 0.50 = random. Faded = earlier versions replaced by the final models; hatched = "
                    "controls with article emotions shuffled. Real-time models also use same-day clicks, so compare within a setting.")
    ax = fig.add_axes([0.36, 0.19, 0.6, 0.64])
    ps.tidy(ax, grid="x")
    y, ticks, labels, pos = 0.0, [], [], []
    for g in dict.fromkeys(d["group"]):
        ax.text(0.5005, y, g, ha="left", va="center", fontsize=15, weight="bold", color=MUTED)
        y -= 1
        for _, r in d[d["group"] == g].iterrows():
            pos.append(y)
            ticks.append(y)
            labels.append(f"{r['name']}  [{r['code']}]")
            y -= 1
        y -= 0.4
    pos = np.array(pos)
    for yy, (_, r) in zip(pos, d.iterrows()):
        col = LIGHT if r["kind"] == "control" else (GREY if r["baseline"] else (ORANGE if r["emotion"] else BLUE))
        ax.barh(yy, r["test_auc"] - 0.5, left=0.5, height=0.72, color=col, alpha=0.45 if r["kind"] == "earlier" else 1,
                hatch="//" if r["kind"] == "control" else None, ec=MUTED if r["kind"] == "control" else "none", lw=0.6)
        ax.text(r["test_auc"] + 0.002, yy, r["label"], va="center", ha="left", fontsize=13, weight="bold" if r["kind"] == "final" else "normal")
    ax.set_yticks(ticks, labels, fontsize=12.5)
    for tl, (_, r) in zip(ax.get_yticklabels(), d.iterrows()):
        tl.set_color(INK if r["kind"] == "final" else MUTED)
    ax.set_xlim(0.5, 0.74)
    ax.set_ylim(y + 0.6, 0.7)
    ax.set_xlabel("AUC on the final test day (0.50 = random)")
    ax.legend(handles=[Patch(color=GREY, label="baseline"), Patch(color=BLUE, label="without emotion"), Patch(color=ORANGE, label="with emotion"),
                       Patch(color=ORANGE, alpha=0.45, label="earlier version"), Patch(fc=LIGHT, ec=MUTED, hatch="//", label="control")],
              loc="upper right", bbox_to_anchor=(0.97, 0.9), bbox_transform=fig.transFigure, ncol=5, frameon=False, fontsize=13)
    ps.save(fig, out / "all_models_auc_every_run.png")
    return d


def all_metrics_heatmap(out):
    m = final_metrics()
    m = m[m["setting"] == STD].set_index("model").loc[STD_ORDER].reset_index()
    d = ps.write_csv(pd.DataFrame({"model": m["model"], "name": m["model"].map(DISPLAY), "family": m["family"],
                                   **{k: m[k].round(3) for k in RANK8}}), out / "all_metrics_heatmap.csv")
    fig = ps.figure("Every ranking metric orders the model families the same way",
                    "Final test day, standard setting. Colour runs from the lowest (light) to the highest (dark) model in each column.")
    ax = fig.add_axes([0.27, 0.13, 0.71, 0.72])
    vals = d[RANK8].to_numpy(float)
    norm = (vals - vals.min(0)) / np.where(np.ptp(vals, 0) > 0, np.ptp(vals, 0), 1)
    ax.imshow(norm, cmap="Blues", vmin=-0.15, vmax=1.15, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=14, color="white" if norm[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(RANK8)), [METRIC_LABEL[k] for k in RANK8], fontsize=14)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(d)), d["name"], fontsize=14)
    for t, e in zip(ax.get_yticklabels(), m["emotion"]):
        t.set_color(VERMILION if e else INK)
    for k in range(1, len(d)):
        if d["family"].iloc[k] != d["family"].iloc[k - 1]:
            ax.axhline(k - 0.5, color="white", lw=4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.035, 0.075, "Names in vermilion use emotion. Within a family, single models can swap places on some metrics.", fontsize=14.5, color=MUTED)
    ps.save(fig, out / "all_metrics_heatmap.png")
    return d


def emotion_gain_all_metrics(out):
    g = gains()
    g = g[g["pair"].isin(STD_PAIRS) & g["metric"].isin(RANK8)].copy()
    g["pair_name"] = g["pair"].map(PAIR_SHORT)
    g["label"] = g["gain"].map(gain_txt)
    g["significant"] = g["significant_holm"].astype(bool)
    d = ps.write_csv(g[["pair", "pair_name", "metric", "metric_name", "gain", "ci_low", "ci_high", "user_ci_low", "user_ci_high", "p_holm",
                        "significant", "dz", "label"]], out / "emotion_gain_all_metrics.csv")
    fig = ps.figure("Standard setting: emotion's gain is positive on every ranking metric, but small",
                    "Gain over the twin, standard setting, final test day, 95% CIs. Light grey, n.s. = not significant (Holm over all tests).")
    names = [PAIR_SHORT[p] for p in STD_PAIRS]
    for k, metric in enumerate(RANK8):
        r, c = divmod(k, 4)
        ax = fig.add_axes([0.2 + c * 0.2, 0.52 - r * 0.38, 0.165, 0.3])
        sub = d[d["metric"] == metric].set_index("pair").loc[STD_PAIRS].reset_index()
        y = np.arange(len(sub))[::-1]
        ax.barh(y, sub["gain"], height=0.66, color=[ps.gain_colour(a, s) for a, s in zip(sub["gain"], sub["significant"])])
        ax.errorbar(sub["gain"], y, xerr=[sub["gain"] - sub["ci_low"], sub["ci_high"] - sub["gain"]], fmt="none", ecolor=INK, elinewidth=1.2,
                    capsize=3)
        ax.axvline(0, color=INK, lw=1)
        ps.tidy(ax, grid="x")
        ax.set_title(METRIC_LABEL[metric], fontsize=16, weight="bold", loc="left")
        ax.set_yticks(y, [f"{n}" if not s else n for n, s in zip(names, sub["significant"])] if c == 0 else [""] * len(y), fontsize=13)
        lo, hi = min(float(sub["ci_low"].min()), 0), max(float(sub["ci_high"].max()), 0)
        span = hi - lo
        ax.set_xlim(lo - 0.1 * span, hi + 0.9 * span)
        ax.locator_params(axis="x", nbins=3)
        ax.tick_params(axis="x", labelsize=11)
        for yy, (_, rr) in zip(y, sub.iterrows()):
            ax.text(max(rr["ci_high"], 0) + 0.05 * span, yy, rr["label"] + ("" if rr["significant"] else " n.s."), va="center", fontsize=12,
                    weight="bold" if rr["significant"] else "normal", color=INK if rr["significant"] else MUTED)
    fig.text(0.035, 0.075, "Axes differ per metric because the metrics have different scales.", fontsize=14, color=MUTED)
    ps.save(fig, out / "emotion_gain_all_metrics.png")
    return d


def emotion_gain_auc(out):
    g = gains()
    pairs = ["Meaning + emotion vs Meaning match", "Tuned profile vs Meaning match", "LightGBM + emotion vs LightGBM (Standard)",
             "LightGBM + emotion vs LightGBM (Real-time)", "Neural + emotion vs Neural"]
    names = {"Meaning + emotion vs Meaning match": "Meaning + emotion", "Tuned profile vs Meaning match": "Meaning + emotion, tuned profile",
             "LightGBM + emotion vs LightGBM (Standard)": "LightGBM + emotion, standard setting",
             "LightGBM + emotion vs LightGBM (Real-time)": "LightGBM + emotion, real-time setting", "Neural + emotion vs Neural": "Neural + emotion"}
    rows = [one(g, "final_twin_gains_all_metrics.csv", pair=p, metric="auc") for p in pairs]
    d = ps.write_csv(pd.DataFrame([{"pair": p, "name": names[p], "setting": r["setting"], "gain": r["gain"], "ci_low": r["ci_low"],
                                    "ci_high": r["ci_high"], "user_ci_low": r["user_ci_low"], "user_ci_high": r["user_ci_high"],
                                    "p_holm": r["p_holm"], "significant": bool(r["significant_holm"]), "label": gain_txt(r["gain"])}
                                   for p, r in zip(pairs, rows)]), out / "emotion_gain_auc.csv")
    ylab = [n + ("" if s else f"\n{NS} ({holm_txt(p)})") for n, s, p in zip(d["name"], d["significant"], d["p_holm"])]
    fig = ps.figure("Emotion adds a small AUC gain; clearest in the standard LightGBM ranker",
                    "AUC gain of each emotion model over its twin without emotion, final test day, 95% CIs. Light grey = not significant (Holm).")
    ax = fig.add_axes([0.33, 0.15, 0.63, 0.7])
    hbar_gains(ax, d, ylab, label_size=16)
    ax.set_xlabel(GAIN_AXIS)
    ps.save(fig, out / "emotion_gain_auc.png")
    return d


def two_settings(out):
    m = final_metrics().set_index("model")
    g = gains()
    rows = []
    for setting, a, b, pair in ((STD, "M4v6_R1", "M5v6_R1", "LightGBM + emotion vs LightGBM (Standard)"),
                                (RT, "M4v6", "M5v6", "LightGBM + emotion vs LightGBM (Real-time)")):
        r = one(g, "final_twin_gains_all_metrics.csv", pair=pair, metric="auc")
        for code in (a, b):
            rows.append({"setting": SETTING[setting], "model": code, "name": DISPLAY[code], "emotion": bool(m.loc[code, "emotion"]),
                         "test_auc": m.loc[code, "auc"], "label": f"{m.loc[code, 'auc']:.4f}", "pair_gain": r["gain"],
                         "gain_label": gain_txt(r["gain"]), "p_holm": r["p_holm"], "significant": bool(r["significant_holm"])})
    d = ps.write_csv(pd.DataFrame(rows), out / "two_settings.csv")
    fig = ps.figure("Emotion adds more when same-day clicks are not available",
                    "Final-test AUC of LightGBM with and without emotion. With same-day clicks, recent clicks already carry most of the signal.")
    ax = fig.add_axes([0.08, 0.2, 0.89, 0.6])
    ps.tidy(ax)
    w = 0.36
    for k, setting in enumerate(d["setting"].unique()):
        t = d[d["setting"] == setting].reset_index(drop=True)
        for j, (_, r) in enumerate(t.iterrows()):
            x = k + (j - 0.5) * w
            ax.bar(x, r["test_auc"], width=w * 0.95, color=ORANGE if r["emotion"] else BLUE)
            ax.text(x, r["test_auc"] + 0.002, r["label"], ha="center", va="bottom", fontsize=17, weight="bold")
        sig = bool(t["significant"].iloc[0])
        note = f"gain {t['gain_label'].iloc[0]}\n" + (holm_txt(t["p_holm"].iloc[0]) if sig else f"{NS} ({holm_txt(t['p_holm'].iloc[0])})")
        ax.text(k, float(t["test_auc"].max()) + 0.011, note, ha="center", va="bottom", fontsize=17, color=INK if sig else MUTED,
                weight="bold" if sig else "normal")
    ax.set_xticks([0, 1], [s.replace(" (", "\n(") for s in d["setting"].unique()], fontsize=17)
    ax.set_ylim(0.6, 0.75)
    ax.set_ylabel(AUC_AXIS.split(":")[0] + " on the final test day")
    ax.legend(handles=[Patch(color=BLUE, label="LightGBM ranker (without emotion)"), Patch(color=ORANGE, label="LightGBM + emotion")],
              loc="upper left", frameon=False)
    ps.save(fig, out / "two_settings.png")
    return d


def two_settings_all_metrics(out):
    g = gains()
    rows = []
    for setting, pair in ((STD, "LightGBM + emotion vs LightGBM (Standard)"), (RT, "LightGBM + emotion vs LightGBM (Real-time)")):
        for k in RANK8:
            r = one(g, "final_twin_gains_all_metrics.csv", pair=pair, metric=k)
            rows.append({"setting": SETTING[setting], "pair": pair, "metric": k, "metric_name": METRIC_LABEL[k], "gain": r["gain"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"], "user_ci_low": r["user_ci_low"], "user_ci_high": r["user_ci_high"],
                         "p_holm": r["p_holm"], "significant": bool(r["significant_holm"]), "label": gain_txt(r["gain"])})
    d = ps.write_csv(pd.DataFrame(rows), out / "two_settings_all_metrics.csv")
    fig = ps.figure("With same-day clicks, emotion makes the top of the list slightly worse",
                    "LightGBM + emotion minus LightGBM on each metric, final test day, 95% CIs, Holm over all 70 tests. The combined ranker "
                    "loses on the same top-5 metrics in the real-time setting (final_twin_gains_all_metrics.csv).")
    axes = [fig.add_axes([0.13, 0.2, 0.34, 0.6]), fig.add_axes([0.63, 0.2, 0.34, 0.6])]
    lo, hi = float(d["ci_low"].min()), float(d["ci_high"].max())
    for ax, setting in zip(axes, dict.fromkeys(d["setting"])):
        t = d[d["setting"] == setting].reset_index(drop=True)
        hbar_gains(ax, t, [n + ("" if s else "\nn.s.") for n, s in zip(t["metric_name"], t["significant"])], label_size=13)
        span = hi - lo
        ax.set_xlim(lo - 0.45 * span, hi + 0.45 * span)
        ax.locator_params(axis="x", nbins=5)
        ax.set_title(setting, fontsize=17, weight="bold", loc="left", pad=12)
        ax.set_xlabel("gain over the twin without emotion")
    fig.legend(handles=[Patch(color=ORANGE, label="significant gain"), Patch(color=VERMILION, label="significant loss"),
                        Patch(color=LIGHT, label=f"{NS} (n.s.)")], loc="upper right", bbox_to_anchor=(0.98, 0.9), ncol=3, frameon=False)
    ps.save(fig, out / "two_settings_all_metrics.png")
    return d


def same_day_check(out):
    m = final_metrics().set_index("model")
    sf = one(tab("v7_single_feature_auc.csv"), "v7_single_feature_auc.csv", feature="ctr_24h")
    early = one(tab("summary.csv"), "summary.csv", model="M4_lightgbm", part="dev")
    rows = [{"panel": "A", "name": "LightGBM ranker\nstandard setting\n(previous days only)", "auc": m.loc["M4v6_R1", "auc"],
             "source": "final_all_metrics.csv M4v6_R1"},
            {"panel": "A", "name": "LightGBM ranker\nreal-time setting\n(with same-day clicks)", "auc": m.loc["M4v6", "auc"],
             "source": "final_all_metrics.csv M4v6"},
            {"panel": "B", "name": "Click rate\nwith same-day clicks", "auc": sf["same_day_auc_R2"], "source": "v7_single_feature_auc.csv (valid day)"},
            {"panel": "B", "name": "Click rate\nprevious days only", "auc": sf["previous_days_auc_R1"], "source": "v7_single_feature_auc.csv (valid day)"}]
    d = ps.write_csv(pd.DataFrame(rows).assign(label=lambda x: x["auc"].map(auc_txt), name=lambda x: x["name"].str.replace("\n", " | "),
                                               earlier_lightgbm_auc=round(float(early["auc"]), 3)), out / "same_day_check.csv")
    fig = ps.figure("LightGBM's big jump in the real-time setting comes from same-day clicks",
                    "A: final-test AUC of LightGBM without emotion. B: one feature alone, the article's 24-hour click rate, validation day. "
                    "Dashed = random.")
    ax1, ax2 = fig.add_axes([0.07, 0.22, 0.45, 0.58]), fig.add_axes([0.62, 0.22, 0.35, 0.58])
    for ax, t, ttl, ylab in ((ax1, d[d["panel"] == "A"], "A. LightGBM ranker, final test day", "AUC (0.50 = random)"),
                             (ax2, d[d["panel"] == "B"], "B. Click rate alone, validation day", "AUC (0.50 = random)")):
        t = t.reset_index(drop=True)
        ps.tidy(ax)
        x = np.arange(len(t))
        ax.bar(x, t["auc"], width=0.55, color=[BLUE if "same-day" in n else SKY for n in t["name"]])
        for i, r in t.iterrows():
            ax.text(i, r["auc"] + 0.004, r["label"], ha="center", va="bottom", fontsize=18, weight="bold")
        ax.axhline(0.5, color=INK, ls="--", lw=1.5)
        ax.set_xticks(x, [n.replace(" | ", "\n") for n in t["name"]], fontsize=14)
        ax.set_ylim(0.45, 0.75)
        ax.set_xlim(-0.6, len(t) - 0.4)
        ax.set_title(ttl, fontsize=17, loc="left", weight="bold", pad=12)
        ax.set_ylabel(ylab)
    ax1.legend(handles=[Patch(color=BLUE, label="uses same-day clicks"), Patch(color=SKY, label="previous days only")], loc="upper left",
               frameon=False)
    fig.text(0.035, 0.085, f"An earlier LightGBM without emotion scored {d['earlier_lightgbm_auc'].iloc[0]:.3f} in the standard setting; "
             "see the full results table.", fontsize=14.5, color=MUTED)
    ps.save(fig, out / "same_day_check.png")
    return d


def fold_evidence(out):
    fe = tab("v7_fold_evidence.csv")
    fm = tab("v7_fold_means.csv").set_index("model")
    rows = []
    for reg, setting, twin in (("R1", STD, "M4v6_R1"), ("R2", RT, "M4v6")):
        t = fe[fe["comparison"] == reg]
        if len(t) != 4:
            raise Missing(f"results/tables/v7_fold_evidence.csv: {reg}")
        for _, r in t.iterrows():
            rows.append({"setting": SETTING[setting], "fold": r["fold"], "gain": r["gain_seed_mean"], "label": gain_txt(r["gain_seed_mean"]),
                         "seed_noise_sd": fm.loc[twin, "seed_sd"]})
    d = ps.write_csv(pd.DataFrame(rows), out / "fold_evidence.csv")
    fig = ps.figure("In the standard setting the emotion gain repeats on every validation fold",
                    "LightGBM + emotion minus LightGBM on four rolling test days (mean of seeds). Shaded: +/-1 seed-noise SD of the twin.")
    ax = fig.add_axes([0.08, 0.17, 0.89, 0.66])
    ps.tidy(ax)
    x = np.arange(4)
    for k, (setting, col, ls, mk) in enumerate(((SETTING[STD], ORANGE, "-", "o"), (SETTING[RT], MUTED, "--", "s"))):
        t = d[d["setting"] == setting].reset_index(drop=True)
        sd = float(t["seed_noise_sd"].iloc[0])
        ax.axhspan(-sd, sd, color=ORANGE if k == 0 else GREY, alpha=0.16 if k == 0 else 0.25, lw=0)
        ax.plot(x, t["gain"], ls=ls, marker=mk, ms=12, lw=3, color=col)
        for i, r in t.iterrows():
            up = k == 0 or i == 3
            ax.text(i + (0.08 if up else -0.08), r["gain"] + (0.0004 if up else -0.0004), r["label"], ha="left" if up else "right",
                    va="bottom" if up else "top", fontsize=15, weight="bold", color=INK if k == 0 else MUTED)
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(x, [f"{f}\n(test day {day} Nov)" for f, day in zip(d["fold"].unique(), (11, 12, 13, 14))], fontsize=15)
    ax.set_xlim(-0.3, 3.6)
    ax.set_ylabel(GAIN_AXIS)
    sds = d.groupby("setting", sort=False)["seed_noise_sd"].first()
    ax.legend(handles=[Line2D([], [], color=ORANGE, marker="o", lw=3, ms=11, label=SETTING[STD]),
                       Line2D([], [], color=MUTED, marker="s", ls="--", lw=3, ms=11, label=SETTING[RT]),
                       Patch(color=ORANGE, alpha=0.3, label=f"seed noise, standard (SD {sds.iloc[0]:.4f})"),
                       Patch(color=GREY, alpha=0.4, label=f"seed noise, real-time (SD {sds.iloc[1]:.4f})")], loc="upper right", frameon=False)
    ps.save(fig, out / "fold_evidence.png")
    return d


def controls(out):
    fm = final_metrics().set_index("model")
    sh = tab("v7_dev_shuffled_controls.csv").set_index("model")
    s = tab("summary.csv")
    rows = []
    for code in ("M4v6_R1", "M5v6_R1", "B0_R1", "B1_R1"):
        if code not in sh.index:
            raise Missing(f"results/tables/v7_dev_shuffled_controls.csv: {code}")
        for kind, v, src in (("real clicks", fm.loc[code, "auc"], "final_all_metrics.csv"),
                             ("clicks shuffled", sh.loc[code, "dev_auc_shuffled_labels"], "v7_dev_shuffled_controls.csv")):
            rows.append({"panel": "A", "model": code, "name": DISPLAY[code], "kind": kind, "emotion": code in ("M5v6_R1", "B1_R1"),
                         "auc": float(v), "label": f"{float(v):.3f}", "source": src})
    for code, name, kind in (("M6_nrms", "Neural ranker (NRMS)", "no emotion"), ("M7b_nrms_emotion_gated", "Neural + emotion\n(earlier version)", "real emotion"),
                             ("M7ctrl_nrms_permuted", "Neural + shuffled emotion\n(control)", "shuffled emotion")):
        v = float(fm.loc[code, "auc"]) if code in fm.index else float(one(s, "summary.csv", model=code, part="dev")["auc"])
        rows.append({"panel": "B", "model": code, "name": name.replace("\n", " | "), "kind": kind, "emotion": code != "M6_nrms", "auc": v,
                     "label": f"{v:.4f}", "source": "final_all_metrics.csv" if code in fm.index else "summary.csv (dev)"})
    d = ps.write_csv(pd.DataFrame(rows), out / "controls.csv")
    fig = ps.figure("Two checks: no leak, and the earlier neural + emotion version gains nothing",
                    "A: final-test AUC of the standard-setting rankers, and the same scores with the clicks shuffled inside each list (must be "
                    "0.50). B: the earlier neural + emotion version scores the same with real and shuffled article emotions; the final Neural + "
                    "emotion passed its shuffled-emotion check on validation (D106).")
    ax1, ax2 = fig.add_axes([0.07, 0.22, 0.5, 0.58]), fig.add_axes([0.66, 0.22, 0.31, 0.58])
    A = d[d["panel"] == "A"]
    names = list(dict.fromkeys(A["name"]))
    w = 0.38
    ps.tidy(ax1)
    for j, kind in enumerate(("real clicks", "clicks shuffled")):
        t = A[A["kind"] == kind].set_index("name").loc[names]
        x = np.arange(len(names)) + (j - 0.5) * w
        cols = [LIGHT if j else (ORANGE if e else BLUE) for e in t["emotion"]]
        ax1.bar(x, t["auc"] - 0.45, bottom=0.45, width=w * 0.95, color=cols, hatch="//" if j else None, ec=MUTED if j else "none", lw=0.6)
        for xx, (_, r) in zip(x, t.iterrows()):
            ax1.text(xx, r["auc"] + 0.003, r["label"], ha="center", va="bottom", fontsize=14, weight="bold")
    ax1.axhline(0.5, color=INK, ls="--", lw=1.5)
    ax1.set_xticks(range(len(names)), [n.replace("Combined ranker (LightGBM + neural)", "Combined ranker").replace(" + ", "\n+ ").replace(" (own", "\n(own") for n in names],
                   fontsize=14)
    ax1.set_ylim(0.45, 0.72)
    ax1.set_ylabel("AUC (0.50 = random)")
    ax1.set_title("A. Shuffled clicks fall to chance", fontsize=17, loc="left", weight="bold", pad=12)
    ax1.legend(handles=[Patch(color=BLUE, label="real clicks"), Patch(fc=LIGHT, ec=MUTED, hatch="//", label="clicks shuffled")],
               loc="upper right", frameon=False)
    B = d[d["panel"] == "B"].reset_index(drop=True)
    ps.tidy(ax2)
    for i, r in B.iterrows():
        ctrl = r["kind"] == "shuffled emotion"
        ax2.bar(i, r["auc"] - 0.6, bottom=0.6, width=0.6, color=LIGHT if ctrl else (ORANGE if r["emotion"] else BLUE),
                hatch="//" if ctrl else None, ec=MUTED if ctrl else "none", lw=0.6)
        ax2.text(i, r["auc"] + 0.0004, r["label"], ha="center", va="bottom", fontsize=15, weight="bold")
    ax2.set_xticks(range(3), [n.replace(" | ", "\n").replace(" (NRMS)", "").replace("Neural + ", "Neural +\n") for n in B["name"]], fontsize=13)
    ax2.set_ylim(0.6, 0.615)
    ax2.set_ylabel("AUC (axis zoomed)")
    ax2.set_title("B. Earlier neural + emotion version", fontsize=17, loc="left", weight="bold", pad=12)
    ps.save(fig, out / "controls.png")
    return d


# ======================================================================= 4_analysis
def dev_sub(split):
    t = tab("v7_subgroups.csv")
    return t[(t["part"] == "dev") & (t["regime"] == "R1") & (t["split"] == split)].copy()


def gain_by_consistency(out):
    t = dev_sub("consistency tertile")
    order = [("low", "Most varied readers"), ("mid", "Middle"), ("high", "Most consistent readers")]
    rows = [one(t, "v7_subgroups.csv", group=g) for g, _ in order]
    d = ps.write_csv(pd.DataFrame([{"group": r["group"], "name": n, "impressions": int(r["impressions"]), "gain": r["gain"], "ci_low": r["ci_low"],
                                    "ci_high": r["ci_high"], "p_holm": r["p_holm"], "significant": bool(r["p_holm"] < 0.05),
                                    "label": gain_txt(r["gain"])} for r, (_, n) in zip(rows, order)]), out / "gain_by_consistency.csv")
    fig = ps.figure("Emotion helps all readers, most of all those with consistent emotional tastes",
                    "LightGBM + emotion minus LightGBM, standard setting, final test day, by how consistent a reader's clicked emotions are. 95% CIs.")
    ax = fig.add_axes([0.08, 0.2, 0.89, 0.63])
    ps.tidy(ax)
    x = np.arange(len(d))
    ax.bar(x, d["gain"], width=0.55, color=[ps.gain_colour(g, s) for g, s in zip(d["gain"], d["significant"])])
    ax.errorbar(x, d["gain"], yerr=[d["gain"] - d["ci_low"], d["ci_high"] - d["gain"]], fmt="none", ecolor=INK, elinewidth=1.8, capsize=8)
    for i, r in d.iterrows():
        ax.text(i, r["ci_high"] + 0.0002, r["label"], ha="center", va="bottom", fontsize=19, weight="bold")
    ax.set_xticks(x, [f"{r['name']}\n({r['impressions']:,} impressions)" for _, r in d.iterrows()], fontsize=16)
    ax.set_ylim(0, float(d["ci_high"].max()) * 1.2)
    ax.set_ylabel(GAIN_AXIS)
    ps.save(fig, out / "gain_by_consistency.png")
    return d


def gain_by_category(out):
    t = dev_sub("clicked category").sort_values("gain", ascending=False)
    d = ps.write_csv(pd.DataFrame({"group": t["group"], "impressions": t["impressions"].astype(int), "gain": t["gain"], "ci_low": t["ci_low"],
                                   "ci_high": t["ci_high"], "p_holm": t["p_holm"], "significant": t["p_holm"] < 0.05,
                                   "label": t["gain"].map(gain_txt)}), out / "gain_by_category.csv")
    names = [f"{r['group']} ({r['impressions']:,})" + ("" if r["significant"] else f"\n{NS}") for _, r in d.iterrows()]
    fig = ps.figure("Gains are largest for TV, news and sports clicks; music clicks lose",
                    "LightGBM + emotion minus LightGBM, standard setting, final test day, by the clicked article's category (impressions).")
    ax = fig.add_axes([0.2, 0.17, 0.76, 0.67])
    hbar_gains(ax, d, names)
    ax.set_xlabel(GAIN_AXIS)
    ax.legend(handles=[Patch(color=ORANGE, label="significant gain"), Patch(color=VERMILION, label="significant loss"),
                       Patch(color=LIGHT, label=NS)], loc="lower right", frameon=False)
    ps.save(fig, out / "gain_by_category.png")
    return d


def not_topic(out):
    a = tab("behaviour_e5_within_category_auc.csv")
    emo = one(a, "behaviour_e5_within_category_auc.csv", group="all", part="valid", score="personal emotion match alone")
    ctl = one(a, "behaviour_e5_within_category_auc.csv", group="all", part="valid", score="labels shuffled (control)")
    g = tab("behaviour_e5_m3_gain_within_category.csv")
    gv = one(g, "behaviour_e5_m3_gain_within_category.csv", group="all", part="valid")
    gd = one(g, "behaviour_e5_m3_gain_within_category.csv", group="all", part="dev")
    rows = [("A", "Emotion match alone", emo, auc_txt), ("A", "Same, labels shuffled\n(control)", ctl, auc_txt),
            ("B", "Validation day\n(14 Nov)", gv, gain_txt), ("B", "Final test day\n(15 Nov)", gd, gain_txt)]
    d = ps.write_csv(pd.DataFrame([{"panel": p, "name": n.replace("\n", " | "), "value": r["mean"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                                    "n": int(r["n"]), "label": f(r["mean"])} for p, n, r, f in rows]).assign(
        significant=lambda x: x["ci_low"] > np.where(x["panel"] == "A", 0.5, 0.0)), out / "not_topic.csv")
    fig = ps.figure("Emotion still ranks clicks among articles of the same category",
                    "Only candidates of one category are compared, pooled. A: AUC vs chance (0.50). B: Meaning + emotion minus Meaning match.")
    ax1, ax2 = fig.add_axes([0.08, 0.2, 0.38, 0.6]), fig.add_axes([0.59, 0.2, 0.38, 0.6])
    A, B = d[d["panel"] == "A"].reset_index(drop=True), d[d["panel"] == "B"].reset_index(drop=True)
    for ax, t, colors, ylim, base, ttl, ylab in (
            (ax1, A, [ORANGE, GREY], (0.45, 0.58), 0.5, "A. Emotion match alone", "AUC (0.50 = random)"),
            (ax2, B, [ps.gain_colour(v, s) for v, s in zip(B["value"], B["significant"])], (-0.001, 0.0065), 0.0,
             "B. Meaning + emotion minus Meaning match", "AUC gain")):
        ps.tidy(ax)
        x = np.arange(len(t))
        ax.bar(x, t["value"] - base, bottom=base, width=0.5, color=colors)
        ax.errorbar(x, t["value"], yerr=[t["value"] - t["ci_low"], t["ci_high"] - t["value"]], fmt="none", ecolor=INK, elinewidth=1.8, capsize=8)
        for i, r in t.iterrows():
            ax.text(i, r["ci_high"] + (ylim[1] - ylim[0]) * 0.02, r["label"] + ("" if r["significant"] or ax is ax1 else f"\n{NS}"),
                    ha="center", va="bottom", fontsize=17, weight="bold")
        ax.axhline(base, color=INK, lw=1.5, ls="--" if base == 0.5 else "-")
        ax.set_xticks(x, [f"{n.replace(' | ', chr(10))}\n({k:,} lists)" for n, k in zip(t["name"], t["n"])], fontsize=14)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(ttl, fontsize=17, loc="left", weight="bold", pad=12)
        ax.set_ylabel(ylab)
    ax1.text(1.48, 0.502, "chance", ha="right", va="bottom", fontsize=14)
    ps.save(fig, out / "not_topic.png")
    return d


def list_quality(out):
    m = tab("final_all_metrics.csv").set_index("model")
    g = gains()
    pairs = [("Meaning", "M2_sbert", "M3_sbert_emotion", "Meaning + emotion vs Meaning match"),
             ("LightGBM", "M4v6_R1", "M5v6_R1", "LightGBM + emotion vs LightGBM (Standard)"),
             ("Neural", "M6_nrms", "M7b_nrms_emotion_gated", "Neural + emotion vs Neural"),
             ("Combined", "B0_R1", "B1_R1_same_weight", "Combined + emotion vs Combined (Standard, same weight)")]
    rows = []
    for metric, title, fmt in (("cat_div@10", "Categories in the top 10", "{:.2f}"), ("emo_div@10", "Emotions in the top 10", "{:.2f}"),
                               ("coverage@10", "Catalogue coverage (% of articles)", "{:.1f}%")):
        for fam, a, b, pair in pairs:
            r = one(g, "final_twin_gains_all_metrics.csv", pair=pair, metric=metric)
            for code, emo in ((a, False), (b, True)):
                v = float(m.loc[code, metric]) * (100 if metric == "coverage@10" else 1)
                rows.append({"metric": metric, "panel": title, "family": fam, "model": code, "name": DISPLAY[code], "emotion": emo, "value": v,
                             "label": fmt.format(v), "gain": r["gain"] * (100 if metric == "coverage@10" else 1),
                             "tested": bool(r["tested"]), "significant": bool(r["significant_holm"])})
    d = ps.write_csv(pd.DataFrame(rows), out / "list_quality.csv")
    fig = ps.figure("Emotion changes the variety of the top 10 by about 1% or less",
                    "Standard setting, final test day: distinct categories and emotions per top 10, and share of test articles reaching any top 10.")
    for k, (metric, panel) in enumerate(d[["metric", "panel"]].drop_duplicates().itertuples(index=False)):
        ax = fig.add_axes([0.06 + k * 0.325, 0.2, 0.27, 0.6])
        ps.tidy(ax)
        t = d[d["metric"] == metric]
        fams = list(dict.fromkeys(t["family"]))
        w = 0.38
        for j, emo in enumerate((False, True)):
            s = t[t["emotion"] == emo].set_index("family").loc[fams]
            x = np.arange(len(fams)) + (j - 0.5) * w
            ax.bar(x, s["value"], width=w * 0.95, color=ORANGE if emo else BLUE)
            for xx, (_, r) in zip(x, s.iterrows()):
                ax.text(xx, r["value"] * 1.01, r["label"], ha="center", va="bottom", fontsize=14, weight="bold", rotation=90)
        ax.set_xticks(np.arange(len(fams)), fams, fontsize=14)
        ax.set_ylim(0, float(t["value"].max()) * 1.38)
        ax.set_title(panel, fontsize=16, weight="bold", loc="left", pad=10)
    fig.legend(handles=[Patch(color=BLUE, label="without emotion"), Patch(color=ORANGE, label="with emotion")], loc="upper right",
               bbox_to_anchor=(0.98, 0.9), ncol=2, frameon=False)
    ps.save(fig, out / "list_quality.png")
    return d


def overview(out):
    panels = ["3_results/all_models_auc.png", "3_results/emotion_gain_auc.png", "3_results/all_metrics_heatmap.png",
              "3_results/two_settings.png", "4_analysis/gain_by_consistency.png", "4_analysis/gain_by_category.png"]
    missing = [p for p in panels if not (out / p).exists()]
    if missing:
        raise Missing("charts not built: " + ", ".join(missing))
    d = ps.write_csv(pd.DataFrame({"position": range(1, 7), "chart": panels}), out / "overview.csv")
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.5, 0.985, "Emotion-aware news ranking: main results", ha="center", va="top", fontsize=24, weight="bold")
    for i, p in enumerate(d["chart"]):
        ax = fig.add_axes([0.005 + (i % 3) * 0.332, 0.475 - (i // 3) * 0.465, 0.325, 0.44])
        ax.imshow(plt.imread(out / p))
        ax.axis("off")
    ps.save(fig, out / "overview.png")
    return d


CHARTS = [("1_method", "pipeline", "How it works", pipeline), ("1_method", "time_split", "How it works", time_split),
          ("1_method", "model_pairs", "Models", model_pairs), ("2_data", "dataset_overview", "Problem", dataset_overview),
          ("2_data", "neutral_fix", "Problem", neutral_fix), ("2_data", "emotion_validation", "How it works", emotion_validation),
          ("2_data", "emotion_match_signal", "Problem", emotion_match_signal),
          ("3_results", "all_models_auc_every_run", "Backup", all_models_auc_every_run),
          ("3_results", "two_settings_all_metrics", "Same-day check", two_settings_all_metrics),
          ("3_results", "same_day_check", "Same-day check", same_day_check), ("3_results", "fold_evidence", "Result 1", fold_evidence),
          ("3_results", "controls", "Backup", controls),
          ("4_analysis", "gain_by_consistency", "Result 2", gain_by_consistency), ("4_analysis", "gain_by_category", "Result 2", gain_by_category),
          ("4_analysis", "not_topic", "Result 2", not_topic), ("4_analysis", "list_quality", "Result 2", list_quality)]
# Superseded by the week-9 set (scripts/08_week9_figures.py, D110) and moved to archive/figures/week9_superseded/: all_models_auc,
# all_models_ndcg10, all_metrics_heatmap, emotion_gain_all_metrics, emotion_gain_auc, two_settings, overview (their functions are kept
# so the archived charts can be rebuilt with --out).
REPORT = {"pipeline": "Methodology: system design", "time_split": "Methodology: evaluation design", "model_pairs": "Methodology: models",
          "dataset_overview": "Data: dataset description", "neutral_fix": "Methodology: emotion labelling",
          "emotion_validation": "Methodology: emotion labelling (validity)", "emotion_match_signal": "Data analysis: is there an emotion signal?",
          "all_models_auc": "Results: main model comparison", "all_models_ndcg10": "Results: main model comparison (second metric)", "all_models_auc_every_run": "Results or appendix: every model run",
          "all_metrics_heatmap": "Results: other ranking metrics", "emotion_gain_all_metrics": "Results: other ranking metrics",
          "emotion_gain_auc": "Results: main finding (emotion vs twin)", "two_settings": "Results: the two settings",
          "two_settings_all_metrics": "Results / discussion: the two settings", "same_day_check": "Discussion: why real-time scores higher",
          "fold_evidence": "Results: robustness", "controls": "Results: validity checks", "gain_by_consistency": "Discussion: who benefits",
          "gain_by_category": "Discussion: where emotion helps", "not_topic": "Discussion: emotion is not just topic",
          "list_quality": "Discussion: list quality (diversity, coverage)", "overview": "Summary / poster"}
SOURCES = {"pipeline": "method; src/config.py (history length)", "time_split": "final_dataset_overview.csv",
           "model_pairs": "names only (src/plot_style.py)", "dataset_overview": "final_dataset_overview.csv",
           "neutral_fix": "story_emotion_mix.csv (checked against emotion_weighted_vs_combined.json, emotion_label_variants_grid.csv, emotion_v2_choice.json)",
           "emotion_validation": "semeval2007_validation.csv", "emotion_match_signal": "emotion_signal_v2.csv",
           "all_models_auc_every_run": "final_all_metrics.csv, summary.csv", "two_settings_all_metrics": "final_twin_gains_all_metrics.csv",
           "controls": "final_all_metrics.csv, v7_dev_shuffled_controls.csv, summary.csv",
           "all_models_auc": "final_all_metrics.csv, summary.csv", "all_models_ndcg10": "final_all_metrics.csv, summary.csv",
           "all_metrics_heatmap": "final_all_metrics.csv", "emotion_gain_all_metrics": "final_twin_gains_all_metrics.csv",
           "emotion_gain_auc": "final_twin_gains_all_metrics.csv", "two_settings": "final_all_metrics.csv, final_twin_gains_all_metrics.csv",
           "same_day_check": "final_all_metrics.csv, v7_single_feature_auc.csv, summary.csv",
           "fold_evidence": "v7_fold_evidence.csv, v7_fold_means.csv", "gain_by_consistency": "v7_subgroups.csv",
           "gain_by_category": "v7_subgroups.csv", "not_topic": "behaviour_e5_within_category_auc.csv, behaviour_e5_m3_gain_within_category.csv",
           "list_quality": "final_all_metrics.csv, final_twin_gains_all_metrics.csv", "overview": "the six chart PNGs"}


def check_labels(name, d):
    """Every label string equals the formatted value in the same CSV row."""
    rules = {"emotion_gain_auc": ("gain", gain_txt), "emotion_gain_all_metrics": ("gain", gain_txt), "gain_by_consistency": ("gain", gain_txt),
             "gain_by_category": ("gain", gain_txt), "fold_evidence": ("gain", gain_txt), "all_models_auc": ("test_auc", auc_txt), "all_models_ndcg10": ("test_ndcg10", auc_txt),
             "same_day_check": ("auc", auc_txt), "two_settings": ("test_auc", lambda v: f"{v:.4f}"),
             "emotion_validation": ("spearman", lambda v: f"{v:.2f}"), "emotion_match_signal": ("dz", lambda v: f"{v:.2f}"),
             "all_models_auc_every_run": ("test_auc", auc_txt), "two_settings_all_metrics": ("gain", gain_txt),
             "neutral_fix": ("pct", lambda v: f"{v:.0f}%")}
    if name == "controls":
        assert all(f"{v:.{3 if p == 'A' else 4}f}" == lab for v, p, lab in zip(d["auc"], d["panel"], d["label"])), "controls: label differs"
    if name in rules:
        col, f = rules[name]
        assert (d[col].map(f) == d["label"]).all(), f"{name}: a label differs from its CSV value"


def takeaways(data):
    t = {"pipeline": "Articles get emotion scores, readers get an emotion profile, and their match is added to the content score.",
         "model_pairs": "Each emotion model is compared with a twin that is identical except for emotion.",
         "overview": "One-page summary of the six main charts."}
    if (d := data.get("time_split")) is not None:
        t["time_split"] = f"Train on {d['impressions'][0]:,} impressions, tune on {d['impressions'][1]:,}, test once on {d['impressions'][2]:,}."
    if (d := data.get("dataset_overview")) is not None:
        t["dataset_overview"] = f"{int(d['articles_total'].iloc[0]):,} articles; the test day alone has {d[(d['name'] == 'Test') & (d['measure'] == 'impressions')]['label'].iloc[0]} impressions."
    if (d := data.get("neutral_fix")) is not None:
        neu = d[d["emotion"] == "neutral"]["label"].tolist()
        t["neutral_fix"] = f"The old rule called {neu[0]} of articles neutral; the fixed rule calls {neu[1]} neutral, so joy, sadness and surprise become visible."
    if (d := data.get("emotion_validation")) is not None:
        go = d[d["source"].str.startswith("GoEmotions")].set_index("emotion")["spearman"]
        t["emotion_validation"] = (f"Our scores agree with human ratings (Spearman {go.min():.2f} to {go.max():.2f}, highest for valence), "
                                   "higher than the NRC word lists on every emotion.")
    if (d := data.get("emotion_match_signal")) is not None:
        p = d[d["feature"] == "cong_pct_cos_personal"].set_index("part")["label"]
        t["emotion_match_signal"] = (f"Before any model: clicked articles match the reader's emotion profile better than skipped ones "
                                     f"(personal match dz {p.iloc[0]} training, {p.iloc[1]} validation).")
    if (d := data.get("all_models_auc")) is not None:
        best = d.loc[d["test_auc"].idxmax()]
        t["all_models_auc"] = f"All final models beat random; the best in the standard setting is {best['name']} at {best['label']}."
    if (d := data.get("all_models_ndcg10")) is not None:
        best = d.loc[d["test_ndcg10"].idxmax()]
        fam = d[d["family"] != "Baselines"].groupby("family")
        g = [s[s["emotion"]]["test_ndcg10"].max() - s[~s["emotion"]]["test_ndcg10"].iloc[0] for _, s in fam]
        t["all_models_ndcg10"] = (f"The second metric, nDCG@10 (order of the top 10): the families rank as with AUC; best is {best['name']} "
                                  f"at {best['label']}; the best emotion model adds {min(g):+.3f} to {max(g):+.3f} per family.")
    if (d := data.get("all_models_auc_every_run")) is not None:
        t["all_models_auc_every_run"] = (f"All {len(d)} runs scored on the test day, including earlier versions and controls, each labelled "
                                         "with its setting and code.")
    if (d := data.get("two_settings_all_metrics")) is not None:
        rt = d[d["setting"] == SETTING[RT]]
        neg = rt[(rt["gain"] < 0) & rt["significant"]]["metric_name"].tolist()
        std = d[d["setting"] == SETTING[STD]]
        first = ("Standard: every metric gains significantly. " if bool(((std["gain"] > 0) & std["significant"]).all())
                 else "Standard: mostly positive. ")
        t["two_settings_all_metrics"] = first + "Real-time: AUC is flat and emotion significantly lowers " + ", ".join(neg) + "."
    if (d := data.get("controls")) is not None:
        a = d[(d["panel"] == "A") & (d["kind"] == "clicks shuffled")]["auc"]
        t["controls"] = (f"With clicks shuffled every ranker scores {a.min():.3f}-{a.max():.3f} (chance); shuffled emotion scores like real "
                         "emotion in the earlier neural + emotion version (the final one passed this check on validation).")
    if "all_metrics_heatmap" in data:
        t["all_metrics_heatmap"] = "MRR, nDCG, precision, recall and hit rate order the model families the same way as AUC."
    if (d := data.get("emotion_gain_all_metrics")) is not None:
        sig = d.groupby("pair_name")["significant"].sum()
        full = [p for p, k in sig.items() if k == len(RANK8)]
        t["emotion_gain_all_metrics"] = ("Standard setting: every gain is positive; significant on all eight metrics for "
                                         + (", ".join(full) if full else "no pair")
                                         + ", mostly not significant beyond AUC for the meaning and neural models.")
    if (d := data.get("emotion_gain_auc")) is not None:
        t["emotion_gain_auc"] = "Emotion adds " + ", ".join(f"{l} ({n.lower()})" for n, l in zip(d["name"], d["label"])) + "."
    if (d := data.get("two_settings")) is not None:
        t["two_settings"] = f"Emotion adds {d['gain_label'][0]} in the standard setting and {d['gain_label'][2]} (not significant) in real time."
    if (d := data.get("same_day_check")) is not None:
        t["same_day_check"] = (f"LightGBM scores {d['label'][1]} with same-day clicks and {d['label'][0]} without; the click rate alone "
                               f"falls from {d['label'][2]} to {d['label'][3]}.")
    if (d := data.get("fold_evidence")) is not None:
        t["fold_evidence"] = "In the standard setting the gain is positive on all four folds; in real time it stays near the noise."
    if (d := data.get("gain_by_consistency")) is not None:
        t["gain_by_consistency"] = f"Positive for all readers, largest for the most consistent ({d['label'].iloc[-1]} vs {d['label'].iloc[0]})."
    if (d := data.get("gain_by_category")) is not None:
        t["gain_by_category"] = f"Largest when the click is {d['group'].iloc[0]} ({d['label'].iloc[0]}); music clicks lose ({d.set_index('group').loc['music', 'label']})."
    if (d := data.get("not_topic")) is not None:
        t["not_topic"] = f"Inside one category the emotion match still beats chance (AUC {d['label'][0]} vs {d['label'][1]}): it is not just topic."
    if (d := data.get("list_quality")) is not None:
        c = d[(d["metric"] == "cat_div@10") & d["emotion"]]
        fall = c[(c["gain"] < 0) & c["significant"]]["family"].tolist()
        t["list_quality"] = ("Emotion changes the variety of the top 10 by about 1% or less"
                             + (f"; category variety falls slightly but significantly for {', '.join(fall)}." if fall else ".")
                             + " Uses the earlier emotion versions of the meaning (M3) and neural (M7b) models.")
    return t


def readme(out, data, skipped):
    tk = takeaways(data)
    lines = ["# Final charts", "",
             "Built by `python scripts/07_final_figures.py` from saved tables only (no training, no dev scoring). Every PNG has a CSV",
             "next to it with the plotted numbers and the exact label text. Sources are in `results/tables/`. Older charts are in",
             "`archive/figures/` (D99). Names and colours: `src/plot_style.py`. Main results use the standard setting (pre-declared",
             "primary); the real-time setting appears only where the two settings are compared. \"Final test\" = 15 Nov (dev).",
             "\"Slide\" = where the chart fits a talk; \"Report section\" = where it fits the written report (D102-D104).", "",
             "| Chart | Slide | Report section | Takeaway | Source |", "|---|---|---|---|---|"]
    for folder, name, slide, _ in CHARTS:
        if name in data:
            path = f"{folder}/{name}.png" if folder else f"{name}.png"
            lines.append(f"| `{path}` | {slide} | {REPORT[name]} | {tk.get(name, '')} | {SOURCES[name]} |")
    old = (out / "README.md").read_text(encoding="utf-8") if (out / "README.md").exists() else ""
    if "<!-- week9:start -->" in old:                       # written by scripts/08_week9_figures.py; kept above the other charts
        block = "<!-- week9:start -->" + old.split("<!-- week9:start -->", 1)[1].split("<!-- week9:end -->", 1)[0] + "<!-- week9:end -->"
        i = lines.index("| Chart | Slide | Report section | Takeaway | Source |")
        lines[i:i] = block.split("\n") + ["", "## Other final charts", ""]
    if skipped:
        lines += ["", "## Not built", ""] + [f"- `{n}`: missing {why}" for n, why in skipped]
    lines += ["", "## EDA (kept as it is)", "",
              "`EDA/` holds the exploratory charts, written by other scripts and not by this one (D102):", "",
              "| Chart | Report section | Written by |", "|---|---|---|",
              "| `EDA/01_part_sizes.png` ... `EDA/06_integrity.png`, `EDA/eda_overview.png`, `EDA/eda_stats.json` | Data: exploratory analysis | `scripts/01_04_eda.py` |",
              "| `EDA/emotion_cross_source_spearman.png` | Methodology: emotion sources agree | `scripts/02_05_diagnostics.py` |",
              "| `EDA/m5_feature_importance.png` | Appendix only: the EARLIER week-4 LightGBM + emotion (M5), not a final model | `scripts/05_03_feature_importance.py` |"]
    lines += ["", "## Names used in charts", "", "| Display name | Code |", "|---|---|"]
    seen = {}
    for code, name in DISPLAY.items():
        seen.setdefault(name, []).append(CODE[code])
    lines += [f"| {n} | {', '.join(dict.fromkeys(c))} |" for n, c in seen.items()]
    lines += ["| Standard setting (previous days only) | R1 |", "| Real-time setting (with same-day clicks) | R2 |", "",
              "## Metrics in simple words", "",
              "- **AUC**: the chance that a clicked article is ranked above a non-clicked one in the same list; 0.50 = random, 1 = perfect.",
              "- **MRR**: 1 divided by the position of the clicked article (1 = first place), averaged over clicks.",
              "- **nDCG@5 / nDCG@10**: how good the order of the top 5 / top 10 is, giving more credit to clicks near the top (0 to 1).",
              "- **Precision@5**: share of the top 5 that the user clicked.",
              "- **Recall@10**: share of the user's clicks that appear in the top 10.",
              "- **Hit rate@5 / @10**: share of lists with at least one click in the top 5 / top 10.",
              "- **Category / emotion diversity@10**: number of different news categories / emotion labels in the top 10.",
              "- **Catalogue coverage@10**: share of all test-day candidate articles that reach at least one top 10.",
              "- **Gain**: emotion model minus its twin without emotion. **Holm p**: p-value corrected for testing many gains at once;",
              "  above 0.05 = not significant. **CI**: 95% confidence interval.", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main(out=FIGURES):
    data, skipped = {}, []
    for folder, name, _, fn in CHARTS:
        d_out = out / folder if folder else out
        d_out.mkdir(parents=True, exist_ok=True)
        try:
            d = fn(out) if name == "overview" else fn(d_out)
        except Missing as e:
            skipped.append((name, str(e)))
            print(f"skipped {name}: missing {e}")
            continue
        check_labels(name, d)
        data[name] = d
        print("wrote", f"{folder}/{name}.png" if folder else f"{name}.png")
    readme(out, data, skipped)
    return data, skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FIGURES))
    main(type(FIGURES)(ap.parse_args().out))
