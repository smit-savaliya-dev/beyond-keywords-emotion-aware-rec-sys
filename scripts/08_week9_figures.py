"""Week-9 figures: one pair per approach (chosen on valid AUC, D109), 8 metrics, emotion gains; built only from saved tables.

Reads:  results/tables/final_metrics_all_models.csv, final_emotion_gains_all_metrics.csv (scripts/08_04_week9_metrics.py).
Writes: results/figures/3_results/01_metrics_table ... 11_emotion_gain_auc (.png + .csv), results/figures/1_method/12_metrics_explained,
        results/figures/00_week9_overview; a copy of each in results/figures/week9/ with its README.md; the week-9 block of
        results/figures/README.md. Colours and names: src/plot_style.py (FAMILY_SHADES).
Run:    python scripts/08_week9_figures.py [--out DIR]
"""
import argparse
import shutil
import textwrap

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from src import plot_style as ps
from src.config import FIGURES, TABLES
from src.metrics import impression_metrics

INK, MUTED = ps.INK, ps.MUTED
METRICS = ["auc", "mrr", "ndcg@5", "ndcg@10", "p@5", "r@10", "hr@5", "hr@10"]
SHORT = {"auc": "AUC", "mrr": "MRR", "ndcg@5": "nDCG@5", "ndcg@10": "nDCG@10", "p@5": "P@5", "r@10": "R@10", "hr@5": "HR@5", "hr@10": "HR@10"}
EXPLAIN = {"auc": "clicked ranked above not-clicked; 0.50 = random", "mrr": "how high the clicks appear (1 / position)",
           "ndcg@5": "quality of the top 5", "ndcg@10": "quality of the top 10", "p@5": "share of top 5 that were clicked",
           "r@10": "share of clicks found in top 10", "hr@5": "at least one click in top 5", "hr@10": "at least one click in top 10"}
AXIS = {"auc": "AUC: chance a clicked article is ranked above a non-clicked one",
        "mrr": "MRR: 1 / position of the clicked article (1 = first place)",
        "ndcg@5": "nDCG@5: quality of the top-5 order (1 = perfect)", "ndcg@10": "nDCG@10: quality of the top-10 order (1 = perfect)",
        "p@5": "Precision@5: share of the top 5 that were clicked", "r@10": "Recall@10: share of the clicks found in the top 10",
        "hr@5": "Hit rate@5: share of lists with a click in the top 5", "hr@10": "Hit rate@10: share of lists with a click in the top 10"}
APPROACHES = ["Baselines", "Meaning", "LightGBM", "Neural", "Combined"]
APPROACH_NAME = {"Baselines": "Baselines", "Meaning": "Meaning match", "LightGBM": "LightGBM", "Neural": "Neural (NRMS)", "Combined": "Combined"}
NS = "not significant"
CREATED = []                                      # (folder, name, takeaway, source) of every figure written


def load():
    m = pd.read_csv(TABLES / "final_metrics_all_models.csv")
    g = pd.read_csv(TABLES / "final_emotion_gains_all_metrics.csv")
    return m, g


def fmt(v, k):
    return f"{v:.3f}"


def gtxt(v):
    return f"{v:+.4f}"


def figure(title, caption, size=(16, 9)):
    assert len(title) <= 74, f"title too long for one line: {title}"
    fig = plt.figure(figsize=size, facecolor="white")
    fig.text(0.03, 0.965, title, fontsize=24, fontweight="bold", va="top", ha="left", color=INK)
    fig.text(0.03, 0.022, textwrap.fill(caption, 128), fontsize=15, color=MUTED, va="bottom", ha="left")
    return fig


def write(fig, out, folder, name, d, takeaway, source):
    path = (out / folder if folder else out)
    path.mkdir(parents=True, exist_ok=True)
    d.to_csv(path / f"{name}.csv", index=False, encoding="utf-8")
    ps.save(fig, path / f"{name}.png")
    CREATED.append((folder, name, takeaway, source))
    print("wrote", f"{folder}/{name}.png" if folder else f"{name}.png", flush=True)
    return d


def family_legend(ax_or_fig, families, loc, anchor=None, ncol=None, fontsize=14):
    handles, labels = [], []
    for f in families:
        light, dark = ps.FAMILY_SHADES[f]
        if f == "Baselines":
            handles.append(Patch(color=light))
            labels.append("Baselines")
        else:
            handles.append((Patch(color=light), Patch(color=dark)))
            labels.append(APPROACH_NAME[f])
    kw = {"handler_map": {tuple: HandlerTuple(ndivide=None, pad=0.15)}, "frameon": False, "fontsize": fontsize,
          "ncol": ncol or len(handles), "loc": loc, "handlelength": 2.6}
    if anchor:
        kw["bbox_to_anchor"] = anchor
    leg = ax_or_fig.legend(handles, labels, **kw)
    fig = ax_or_fig if hasattr(ax_or_fig, "add_axes") else ax_or_fig.figure
    fig.text(0.97, 0.845, "light shade = without emotion, dark shade = with emotion", ha="right", va="top", fontsize=13, color=MUTED)
    return leg


def std_models(m):
    s = m[m["setting"] == "Standard"].copy()
    s["order"] = s["approach"].map(APPROACHES.index)
    return s.sort_values(["order", "emotion"], kind="mergesort").reset_index(drop=True)


def positions(s):
    """x position of each model: pairs side by side, a gap between approaches."""
    x, cur, prev = [], 0.0, None
    for app in s["approach"]:
        if prev is not None:
            cur += 1.0 if app == prev else 1.6
        x.append(cur)
        prev = app
    return np.array(x)


def wrap_name(n, width=10):
    return "\n".join(textwrap.wrap(n, width))


# ---------------------------------------------------------------------------------------------- 03-09: one bar chart per metric
def bars(ax, m, g, k, ylim, ref=None, gain_row=True, label_size=14, name_size=13):
    s = std_models(m)
    x = positions(s)
    cols = [ps.shade(a, e) for a, e in zip(s["approach"], s["emotion"])]
    ax.bar(x, s[k], width=0.84, color=cols, edgecolor="white", linewidth=2, zorder=2)
    span = ylim[1] - ylim[0]
    labels = []
    for xi, v, e, a in zip(x, s[k], s["emotion"], s["approach"]):
        lab = fmt(v, k)
        labels.append(lab)
        ax.text(xi, v + span * 0.012, lab, ha="center", va="bottom", fontsize=label_size, color=INK, zorder=3)
        if e:
            ax.text(xi, ylim[0] + span * 0.04, "+ emotion", ha="center", va="bottom", fontsize=11.5, color="white", weight="bold",
                    rotation=90, zorder=3)
    if ref is not None:
        ax.axhline(ref[0], color=MUTED, ls="--", lw=1.6, zorder=1)
    gains = {}
    if gain_row:
        gs = g[(g["setting"] == "Standard") & (g["metric"] == k)].set_index("approach")
        for app in gs.index:
            idx = np.flatnonzero(s["approach"].to_numpy() == app)
            top = s.loc[idx, k].max()
            r = gs.loc[app]
            t = f"emotion {gtxt(r['gain'])}" + ("" if r["significant_holm"] else f"\n({NS})")
            gains[app] = t
            ax.text(x[idx].mean(), top + span * 0.075, t, ha="center", va="bottom", fontsize=13,
                    color=INK if r["significant_holm"] else MUTED, weight="bold" if r["significant_holm"] else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_name(n) for n in s["display_name"]], fontsize=name_size)
    ax.set_ylim(*ylim)
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    ps.tidy(ax)
    return s.assign(x=x, label=labels, colour=cols, gain_text=s["approach"].map(gains).fillna(""))


def axis_floor(vals, step=0.05):
    return float(np.floor((min(vals) * 0.92) / step) * step)


def metric_chart(out, m, g, k, name, title, takeaway, ref=None):
    s = std_models(m)
    lo = 0.5 if k == "auc" else axis_floor(s[k])
    hi = float(s[k].max()) + (float(s[k].max()) - lo) * 0.32
    cap = (f"What this shows: {SHORT[k]} on the test day (15 Nov) for every final model, standard setting (previous days only). "
           f"Above each pair: the gain from emotion (Holm-corrected over 32 tests). Take-away: {takeaway}")
    fig = figure(title, cap)
    ax = fig.add_axes([0.085, 0.235, 0.895, 0.575])
    d = bars(ax, m, g, k, (lo, hi), ref)
    ax.set_ylabel(textwrap.fill(AXIS[k], 34), fontsize=15)
    fig.text(0.085, 0.82, f"axis starts at {lo:.2f}" + (" (dashed line = random)" if ref else ""), fontsize=13, color=MUTED, style="italic")
    family_legend(fig, APPROACHES, "upper left", (0.025, 0.905), ncol=5, fontsize=14)
    d = d[["display_name", "approach", "emotion", k, "label", "gain_text"]].rename(columns={k: "value"}).assign(metric=k, axis_from=lo)
    assert (d["value"].map(lambda v: fmt(v, k)) == d["label"]).all()
    return write(fig, out, "3_results", name, d, takeaway, "final_metrics_all_models.csv, final_emotion_gains_all_metrics.csv")


def takeaway_for(m, g, k):
    s = std_models(m)
    best = s.loc[s[k].idxmax()]
    gs = g[(g["setting"] == "Standard") & (g["metric"] == k)]
    sig = gs[gs["significant_holm"] & (gs["gain"] > 0)]["approach"].map(APPROACH_NAME).tolist()
    ns = gs[~gs["significant_holm"]]["approach"].map(APPROACH_NAME).tolist()
    t = f"best is {best['display_name']} ({fmt(best[k], k)}); emotion helps significantly for " + (", ".join(sig) if sig else "no approach")
    return t + (f"; {', '.join(ns)}: {NS}." if ns else ".")


def hit_rate(out, m, g):
    s = std_models(m)
    best = s.loc[s["hr@10"].idxmax()]
    tk = f"Hit rate@5: {takeaway_for(m, g, 'hr@5')} Hit rate@10: {takeaway_for(m, g, 'hr@10')}"
    fig = figure(f"A click in the top 5 / top 10 (hit rate): best is {best['display_name']}",
                 "What this shows: share of test-day lists with at least one click in the top 5 (left) and in the top 10 (right), standard "
                 "setting; the gain from emotion is written next to each emotion model. Take-away: LightGBM and Combined lead; emotion adds to every learned ranker.")
    y = -positions(s)
    cols = [ps.shade(a, e) for a, e in zip(s["approach"], s["emotion"])]
    parts = []
    for i, k in enumerate(["hr@5", "hr@10"]):
        ax = fig.add_axes([0.21 + i * 0.4, 0.15, 0.33, 0.62])
        ax.barh(y, s[k], height=0.84, color=cols, edgecolor="white", linewidth=2, zorder=2)
        gs = g[(g["setting"] == "Standard") & (g["metric"] == k)].set_index("approach")
        labels, gt = [], []
        for yi, v, a, e in zip(y, s[k], s["approach"], s["emotion"]):
            lab = f"{v:.3f}"
            labels.append(lab)
            t, sig = "", True
            if e:
                r = gs.loc[a]
                sig = bool(r["significant_holm"])
                t = f"  ({gtxt(r['gain'])}" + ("" if sig else f", {NS}") + ")"
            gt.append(t.strip())
            ax.text(v + 0.006, yi, lab + t, va="center", fontsize=13.5, color=INK if sig else MUTED,
                    weight="bold" if e and sig else "normal")
        ax.set_xlim(0.4, 0.95)
        ax.set_yticks(y)
        ax.set_yticklabels(s["display_name"] if i == 0 else [""] * len(s), fontsize=14)
        ax.set_title(AXIS[k], fontsize=15, loc="left", pad=10)
        ax.set_xlabel("share of lists (axis starts at 0.40)", fontsize=14)
        ps.tidy(ax, grid="x")
        parts.append(s[["display_name", "approach", "emotion", k]].rename(columns={k: "value"}).assign(metric=k, label=labels, gain_text=gt))
    family_legend(fig, APPROACHES, "upper left", (0.025, 0.905), ncol=5, fontsize=14)
    d = pd.concat(parts)[["metric", "display_name", "approach", "emotion", "value", "label", "gain_text"]]
    assert (d["value"].map(lambda v: f"{v:.3f}") == d["label"]).all()
    return write(fig, out, "3_results", "09_hitrate", d, tk, "final_metrics_all_models.csv, final_emotion_gains_all_metrics.csv")


# ---------------------------------------------------------------------------------------------- 01: metrics table
def metrics_table(out, m, g):
    s = std_models(m)
    best = {k: s[k].max() for k in METRICS}
    fig = figure("The learned rankers lead on all 8 metrics; emotion lifts every pair",
                 "What this shows: every final model (rows, grouped by approach; without and with emotion next to each other) on 8 test-day "
                 "metrics, standard setting. Darker cell = better within the column; bold = best in the column. Take-away: LightGBM and Combined "
                 "lead everywhere (they swap places on R@10 / HR@10); each emotion model is above its twin in every column.")
    ax = fig.add_axes([0.02, 0.135, 0.96, 0.755])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    name_w, col_w = 28.0, 72.0 / len(METRICS)
    head_h, expl_h = 6.0, 11.0
    row_h = (100 - head_h - expl_h) / len(s)
    cmap = LinearSegmentedColormap.from_list("t", list(ps.TABLE_RAMP))
    y = 100 - head_h
    for j, k in enumerate(METRICS):
        xc = name_w + col_w * (j + 0.5)
        ax.text(xc, 100 - head_h / 2, SHORT[k], ha="center", va="center", fontsize=18, weight="bold")
        ax.text(xc, y - expl_h / 2, textwrap.fill(EXPLAIN[k], 15), ha="center", va="center", fontsize=12.5, color=MUTED, linespacing=1.1)
    ax.text(1.0, 100 - head_h / 2, "Model", ha="left", va="center", fontsize=18, weight="bold")
    ax.text(1.0, y - expl_h / 2, "test day, 15 Nov\n(higher = better)", ha="left", va="center", fontsize=12.5, color=MUTED)
    ax.plot([0, 100], [y - expl_h, y - expl_h], color=INK, lw=1.5)
    rows = []
    top = y - expl_h
    prev = None
    for i, r in s.iterrows():
        y0 = top - (i + 1) * row_h
        if prev is not None and r["approach"] != prev:
            ax.plot([0, 100], [y0 + row_h, y0 + row_h], color=MUTED, lw=1.2)
        prev = r["approach"]
        ax.add_patch(Rectangle((0.6, y0 + row_h * 0.18), 1.8, row_h * 0.64, color=ps.shade(r["approach"], r["emotion"])))
        ax.text(3.2, y0 + row_h / 2, r["display_name"], ha="left", va="center", fontsize=15.5, weight="bold" if r["emotion"] else "normal")
        for j, k in enumerate(METRICS):
            lo, hi = s[k].min(), s[k].max()
            frac = (r[k] - lo) / (hi - lo) if hi > lo else 1.0
            ax.add_patch(Rectangle((name_w + col_w * j + 0.15, y0 + 0.12), col_w - 0.3, row_h - 0.24, color=cmap(frac), lw=0))
            lab = f"{r[k]:.3f}"
            ax.text(name_w + col_w * (j + 0.5), y0 + row_h / 2, lab, ha="center", va="center", fontsize=16,
                    weight="bold" if r[k] == best[k] else "normal")
            rows.append({"model": r["display_name"], "approach": r["approach"], "emotion": r["emotion"], "metric": k, "value": r[k],
                         "label": lab, "shade_fraction": frac, "best_in_column": bool(r[k] == best[k])})
    d = pd.DataFrame(rows)
    assert (d["value"].map(lambda v: f"{v:.3f}") == d["label"]).all()
    bm = d[d["best_in_column"]].groupby("model")["metric"].count()
    above = all(s[s["approach"] == a].sort_values("emotion")[k].diff().iloc[-1] > 0 for a in APPROACHES[1:] for k in METRICS)
    tk = (f"{bm.idxmax()} is best in {bm.max()} of 8 columns; "
          + ("every emotion model is above its twin in every column." if above else "emotion models are mostly above their twins."))
    return write(fig, out, "3_results", "01_metrics_table", d, tk, "final_metrics_all_models.csv")


# ---------------------------------------------------------------------------------------------- 02: emotion gain table
def gain_table(out, m, g):
    gs = g[g["setting"] == "Standard"].copy()
    gs["order"] = gs["approach"].map(APPROACHES.index)
    pairs = gs.sort_values("order")["approach"].unique()
    n_sig = int((gs["significant_holm"] & (gs["gain"] > 0)).sum())
    fig = figure(f"Emotion helps on {n_sig} of 32 approach x metric checks, harms none",
                 "What this shows: gain from adding emotion (emotion model minus its twin) on each test-day metric, standard setting. "
                 "Green = significant gain, red = significant loss, grey = not significant (Holm-corrected over all 32 tests, 95% CI by user). "
                 "Take-away: the gain is small but consistent.")
    ax = fig.add_axes([0.02, 0.165, 0.96, 0.72])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    name_w, col_w = 24.0, 76.0 / len(METRICS)
    head_h, expl_h = 8.0, 14.0
    row_h = (100 - head_h - expl_h) / len(pairs)
    y = 100 - head_h
    for j, k in enumerate(METRICS):
        xc = name_w + col_w * (j + 0.5)
        ax.text(xc, 100 - head_h / 2, SHORT[k], ha="center", va="center", fontsize=18, weight="bold")
        ax.text(xc, y - expl_h / 2, textwrap.fill(EXPLAIN[k], 15), ha="center", va="center", fontsize=12.5, color=MUTED, linespacing=1.1)
    ax.text(1.0, 100 - head_h / 2, "Approach", ha="left", va="center", fontsize=18, weight="bold")
    ax.text(1.0, y - expl_h / 2, "with emotion minus\nthe same model without", ha="left", va="center", fontsize=12.5, color=MUTED)
    rows = []
    top = y - expl_h
    for i, app in enumerate(pairs):
        y0 = top - (i + 1) * row_h
        light, dark = ps.FAMILY_SHADES[app]
        ax.add_patch(Rectangle((0.6, y0 + row_h * 0.3), 1.0, row_h * 0.4, color=light))
        ax.add_patch(Rectangle((1.6, y0 + row_h * 0.3), 1.0, row_h * 0.4, color=dark))
        r0 = gs[gs["approach"] == app].iloc[0]
        ax.text(3.4, y0 + row_h * 0.62, APPROACH_NAME[app], ha="left", va="center", fontsize=17, weight="bold")
        ax.text(3.4, y0 + row_h * 0.3, textwrap.fill(f"{r0['emotion_name']} vs {r0['twin_name']}", 30), ha="left", va="center",
                fontsize=12, color=MUTED)
        for j, k in enumerate(METRICS):
            r = gs[(gs["approach"] == app) & (gs["metric"] == k)].iloc[0]
            sig = bool(r["significant_holm"])
            face = ("#cfe9d8" if r["gain"] > 0 else "#f3c9c4") if sig else "#e6e6e6"
            ax.add_patch(Rectangle((name_w + col_w * j + 0.2, y0 + 0.3), col_w - 0.4, row_h - 0.6, color=face, lw=0))
            lab = gtxt(r["gain"])
            sub = ("Holm p < 0.001" if r["p_holm"] < 0.001 else f"Holm p = {r['p_holm']:.3f}") if sig else "n.s."
            ax.text(name_w + col_w * (j + 0.5), y0 + row_h * 0.6, lab, ha="center", va="center", fontsize=17,
                    color=INK if sig else MUTED, weight="bold" if sig else "normal")
            ax.text(name_w + col_w * (j + 0.5), y0 + row_h * 0.3, sub, ha="center", va="center", fontsize=12, color=MUTED)
            rows.append({"approach": app, "metric": k, "gain": r["gain"], "label": lab, "p_holm": r["p_holm"], "significant": sig,
                         "sub_label": sub, "cell": "gain" if sig and r["gain"] > 0 else ("loss" if sig else "n.s.")})
    ax.legend(handles=[Patch(color="#cfe9d8", label="significant gain"), Patch(color="#f3c9c4", label="significant loss"),
                       Patch(color="#e6e6e6", label="not significant (n.s.)")], loc="lower right", bbox_to_anchor=(1.0, -0.1),
              ncol=3, frameon=False, fontsize=14)
    d = pd.DataFrame(rows)
    assert (d["gain"].map(gtxt) == d["label"]).all()
    ns = d[~d["significant"]]
    tk = (f"{n_sig} of 32 gains are significant and positive, none is a significant loss; the not-significant ones are "
          + ", ".join(f"{APPROACH_NAME[a]} {SHORT[k]}" for a, k in zip(ns["approach"], ns["metric"])) + ".")
    return write(fig, out, "3_results", "02_emotion_gain_table", d, tk, "final_emotion_gains_all_metrics.csv")


# ---------------------------------------------------------------------------------------------- 10: two settings
def two_settings(out, m, g):
    l = m[m["approach"] == "LightGBM"].copy()
    l["order"] = l["setting"].map({"Standard": 0, "Real-time": 1})
    l = l.sort_values(["order", "emotion"]).reset_index(drop=True)
    x = np.array([0, 1, 2.8, 3.8])
    gr = g[(g["approach"] == "LightGBM") & (g["metric"] == "auc")].set_index("setting")
    st, rt = gr.loc["Standard"], gr.loc["Real-time"]
    fig = figure("Emotion helps LightGBM on previous-day data, not with same-day clicks",
                 f"What this shows: test-day AUC of LightGBM without and with emotion in two settings. Standard = features from previous days "
                 f"only (the main setting); real-time = also uses earlier clicks of the same day. Take-away: same-day clicks lift AUC by about "
                 f"{l.loc[2, 'auc'] - l.loc[0, 'auc']:.2f}; on top of them emotion adds nothing significant.")
    ax = fig.add_axes([0.09, 0.3, 0.62, 0.52])
    cols = [ps.shade("LightGBM", e) for e in l["emotion"]]
    ax.bar(x, l["auc"], width=0.84, color=cols, edgecolor="white", linewidth=2, zorder=2)
    lo, hi = 0.5, 0.75
    labels = []
    for xi, v, e in zip(x, l["auc"], l["emotion"]):
        labels.append(f"{v:.4f}")
        ax.text(xi, v + 0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=16)
        if e:
            ax.text(xi, lo + 0.01, "+ emotion", ha="center", va="bottom", fontsize=13, color="white", weight="bold", rotation=90)
    gt = []
    for (a, b), r in zip(((0, 1), (2, 3)), (st, rt)):
        t = f"emotion {gtxt(r['gain'])}" + ("" if r["significant_holm"] else f"\n({NS})")
        gt.append(t)
        ax.text((x[a] + x[b]) / 2, l["auc"].iloc[[a, b]].max() + 0.02, t, ha="center", va="bottom", fontsize=16,
                weight="bold" if r["significant_holm"] else "normal", color=INK if r["significant_holm"] else MUTED)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["LightGBM\nranker", "LightGBM\n+ emotion"] * 2, fontsize=15)
    for xc, t in ((0.5, "previous days only\n(standard setting)"), (3.3, "with same-day clicks\n(real-time setting)")):
        ax.text(xc, -0.17, t, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=16, weight="bold")
    ax.set_ylim(lo, hi)
    ax.set_ylabel(textwrap.fill(AXIS["auc"], 34), fontsize=15)
    ps.tidy(ax)
    fig.text(0.09, 0.835, "axis starts at 0.50 (= random)", fontsize=13, color=MUTED, style="italic")
    fig.legend(handles=[Patch(color=cols[0], label="LightGBM without emotion"), Patch(color=cols[1], label="LightGBM + emotion")],
               loc="upper left", bbox_to_anchor=(0.74, 0.8), frameon=False, fontsize=15)
    fig.text(0.74, 0.62, textwrap.fill(f"Standard: gain {gtxt(st['gain'])}, 95% CI {gtxt(st['ci_low'])} to {gtxt(st['ci_high'])}, "
                                       f"Holm p {st['p_holm']:.1e}.", 30), fontsize=14, va="top")
    fig.text(0.74, 0.47, textwrap.fill(f"Real-time: gain {gtxt(rt['gain'])}, 95% CI {gtxt(rt['ci_low'])} to {gtxt(rt['ci_high'])}, "
                                       f"Holm p {rt['p_holm']:.3f} ({NS}).", 30), fontsize=14, va="top", color=MUTED)
    d = l[["display_name", "setting", "emotion", "auc"]].assign(label=labels, pair_gain_text=[gt[0], gt[0], gt[1], gt[1]])
    tk = f"Emotion adds {gtxt(st['gain'])} AUC in the standard setting (significant) and {gtxt(rt['gain'])} with same-day clicks ({NS})."
    return write(fig, out, "3_results", "10_two_settings", d, tk, "final_metrics_all_models.csv, final_emotion_gains_all_metrics.csv")


# ---------------------------------------------------------------------------------------------- 11: AUC gain per approach
def gain_auc(out, m, g):
    gs = g[(g["setting"] == "Standard") & (g["metric"] == "auc")].copy()
    gs["order"] = gs["approach"].map(APPROACHES.index)
    gs = gs.sort_values("order", ascending=False).reset_index(drop=True)
    fig = figure(f"Emotion raises AUC significantly for every approach ({gtxt(gs['gain'].min())} to {gtxt(gs['gain'].max())})",
                 "What this shows: AUC gain from emotion (emotion model minus its twin) on the test day, standard setting; line = 95% "
                 "confidence interval (clustered by user). Grey would mean not significant (Holm). Take-away: every gain is small but above zero; "
                 "the intervals overlap, so the approaches do not differ clearly from each other.")
    ax = fig.add_axes([0.3, 0.17, 0.64, 0.62])
    y = np.arange(len(gs))
    cols = [ps.FAMILY_SHADES[a][1] if s else ps.NS_GREY for a, s in zip(gs["approach"], gs["significant_holm"])]
    ax.barh(y, gs["gain"], height=0.6, color=cols, zorder=2)
    ax.errorbar(gs["gain"], y, xerr=[gs["gain"] - gs["ci_low"], gs["ci_high"] - gs["gain"]], fmt="none", ecolor=INK, capsize=8, lw=2, zorder=3)
    labels = []
    for yi, r in gs.iterrows():
        lab = gtxt(r["gain"]) + ("" if r["significant_holm"] else f" ({NS})")
        labels.append(lab)
        ax.text(r["ci_high"] + 0.0002, yi, lab, va="center", fontsize=17, weight="bold")
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{e}\nvs {t}" for e, t in zip(gs["emotion_name"], gs["twin_name"])], fontsize=15)
    ax.set_xlim(0, float(gs["ci_high"].max()) * 1.45)
    ax.set_xlabel("AUC gain from emotion (higher = emotion helps more; 0 = no effect)", fontsize=15)
    ps.tidy(ax, grid="x")
    family_legend(fig, APPROACHES[1:], "upper left", (0.025, 0.905), ncol=4, fontsize=14)
    d = gs[["approach", "emotion_name", "twin_name", "gain", "ci_low", "ci_high", "p", "p_holm", "dz", "significant_holm"]].assign(label=labels)
    tk = ("Emotion adds " + ", ".join(f"{gtxt(r['gain'])} ({APPROACH_NAME[r['approach']]})" for _, r in gs[::-1].iterrows())
          + " AUC; all significant; the 95% intervals overlap, so no approach clearly gains more than another.")
    return write(fig, out, "3_results", "11_emotion_gain_auc", d, tk, "final_emotion_gains_all_metrics.csv")


# ---------------------------------------------------------------------------------------------- 12: metrics explained
def metrics_explained(out):
    clicked = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0], dtype=float)           # 10 articles, clicks at positions 2 and 7
    scores = np.arange(10, 0, -1, dtype=float)                                  # already in ranked order
    v = impression_metrics(clicked, scores, (5, 10), np.random.default_rng(0))
    fig = figure("How to read the metrics: one example list of 10 articles with 2 clicks",
                 "What this shows: a ranked list (left) and what each metric says about it (right), computed with the project's own metric "
                 "code. Take-away: AUC looks at the whole list, the @5 / @10 metrics only at the top of it.")
    ax = fig.add_axes([0.03, 0.1, 0.94, 0.78])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    box_h = 8.2
    for i in range(10):
        y0 = 92 - (i + 1) * box_h - i * 0.6 + 6
        c = clicked[i] == 1
        ax.add_patch(FancyBboxPatch((6, y0), 27.5, box_h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="#cfe9d8" if c else "#f2f2f2", ec=ps.GAIN_GREEN if c else "#bdbdbd", lw=2 if c else 1))
        ax.text(3.8, y0 + box_h / 2, f"{i + 1}", ha="right", va="center", fontsize=16, weight="bold")
        ax.text(8, y0 + box_h / 2, f"Article {chr(65 + i)}" + (f"  clicked  ->  MRR 1/{i + 1}" if c else ""), ha="left", va="center", fontsize=15,
                weight="bold" if c else "normal")
    ax.text(3.8, 97, "rank", ha="right", va="center", fontsize=13, color=MUTED)
    y5 = 92 - 5 * box_h - 4 * 0.6 + 6
    y10 = 92 - 10 * box_h - 9 * 0.6 + 6
    for yb, lab in ((y5, "top 5"), (y10, "top 10")):
        ax.plot([34, 35.5, 35.5, 34], [98, 98, yb, yb], color=MUTED, lw=1.8)
        ax.text(36.2, (98 + yb) / 2 if lab == "top 5" else yb + 3, lab, fontsize=14, color=MUTED, va="center")
    cards = [("AUC", "auc", "Of the 2 x 8 = 16 (clicked, not clicked) pairs, how many have the clicked article higher? 0.50 = random, 1 = perfect."),
             ("MRR", "mrr", "1 / position of each click, averaged: (1/2 + 1/7) / 2. Rewards a click near the top."),
             ("nDCG@10", "ndcg@10", "Quality of the top-10 order: clicks count more the higher they are (1 = both clicks at positions 1 and 2)."),
             ("P@5", "p@5", "Share of the top 5 that was clicked: 1 click in 5 articles."),
             ("R@10", "r@10", "Share of the clicks found in the top 10: both clicks are there."),
             ("HR@10", "hr@10", "Is there at least one click in the top 10? Yes = 1, no = 0; averaged over lists it is a share.")]
    rows = []
    for i, (name, k, text) in enumerate(cards):
        y0 = 84 - i * 15.2
        ax.add_patch(FancyBboxPatch((42, y0), 57, 13.4, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec="#bdbdbd", lw=1.2))
        lab = f"{v[k]:.3f}" if k in ("auc", "mrr", "ndcg@10") else f"{v[k]:.2f}"
        ax.text(43.5, y0 + 9.2, f"{name} = {lab}", fontsize=18, weight="bold", va="center")
        ax.text(43.5, y0 + 3.9, textwrap.fill(text, 78), fontsize=13.5, va="center", color=INK, linespacing=1.15)
        rows.append({"metric": name, "key": k, "value": v[k], "label": lab, "explanation": text})
    d = pd.DataFrame(rows).assign(example_clicks="positions 2 and 7 of 10")
    tk = "A worked example (10 articles, clicks at positions 2 and 7) showing what AUC, MRR, nDCG@10, P@5, R@10 and HR@10 measure."
    return write(fig, out, "1_method", "12_metrics_explained", d, tk, "computed with src/metrics.py on the example list")


# ---------------------------------------------------------------------------------------------- 00: overview
def overview(out):
    panels = ["3_results/01_metrics_table.png", "3_results/02_emotion_gain_table.png", "3_results/03_auc.png",
              "3_results/06_ndcg10.png", "3_results/10_two_settings.png", "3_results/11_emotion_gain_auc.png"]
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.5, 0.985, "Emotion-aware news ranking: week-9 results (test day, standard setting)", ha="center", va="top", fontsize=24,
             weight="bold")
    for i, p in enumerate(panels):
        ax = fig.add_axes([0.005 + (i % 3) * 0.332, 0.475 - (i // 3) * 0.465, 0.325, 0.44])
        ax.imshow(plt.imread(out / p))
        ax.axis("off")
    d = pd.DataFrame({"position": range(1, 7), "chart": panels})
    return write(fig, out, "", "00_week9_overview", d, "One 16:9 page with figures 01, 02, 03, 06, 10 and 11.", "the six chart PNGs")


# ---------------------------------------------------------------------------------------------- READMEs
NAMES_TABLE = [("Popularity", "M0_popularity"), ("Keyword match (TF-IDF)", "M1_tfidf"), ("Meaning match (Sentence-BERT)", "M2_sbert"),
               ("Meaning + emotion", "M3b_plus (tuned profile)"), ("LightGBM ranker", "M4v6_R1 (standard); M4v6 (real-time, figure 10)"),
               ("LightGBM + emotion", "M5v6_R1 (standard); M5v6 (real-time, figure 10)"), ("Neural ranker (NRMS)", "M6_nrms"),
               ("Neural + emotion", "M7d_nrms_emotion_fusion"), ("Combined ranker", "B0_R1"),
               ("Combined + emotion", "B1_R1_same_weight (B1_R1 at B0_R1's blend weight)"),
               ("previous days only (standard setting)", "R1"), ("with same-day clicks (real-time setting)", "R2")]
GLOSSARY = ["- **AUC**: chance that a clicked article is ranked above a non-clicked one in the same list; 0.50 = random, 1 = perfect.",
            "- **MRR**: 1 / position of the clicked article (1 = first place), averaged over clicks: how high the clicks appear.",
            "- **nDCG@5 / nDCG@10**: quality of the order of the top 5 / top 10, with more credit for clicks near the top (0 to 1).",
            "- **P@5 (Precision@5)**: share of the top 5 that were clicked.",
            "- **R@10 (Recall@10)**: share of the user's clicks found in the top 10.",
            "- **HR@5 / HR@10 (Hit rate)**: share of lists with at least one click in the top 5 / top 10.",
            "- **Gain**: emotion model minus its twin without emotion, on the same test-day lists. **95% CI**: clustered by user.",
            "  **Holm p**: paired t-test p corrected for the 32 tests (4 approaches x 8 metrics); above 0.05 = not significant (n.s.)."]


def readmes(out):
    rows = [(f"{f}/{n}.png" if f else f"{n}.png", n, tk, src) for f, n, tk, src in CREATED]
    order = sorted(rows, key=lambda r: r[1])
    lines = ["# Week-9 figures", "",
             "Copies of every figure made or updated by `python scripts/08_week9_figures.py` (originals in `3_results/`, `1_method/` and the",
             "figures root). Built only from saved tables (`scripts/08_04_week9_metrics.py`), test day = 15 Nov, standard setting unless",
             "stated. Models: one per approach, chosen on validation AUC (D109). PNGs only here; the CSV with the plotted numbers sits next",
             "to each original.", "",
             "| Figure | Takeaway | Source CSV |", "|---|---|---|"]
    lines += [f"| `{n}.png` | {tk} | {src} |" for _, n, tk, src in order]
    lines += ["", "## Names used", "", "| Display name | Internal code |", "|---|---|"] + [f"| {a} | {b} |" for a, b in NAMES_TABLE]
    lines += ["", "Every earlier version and its validation and test-day scores: `results/tables/all_versions_full.csv` (not in figures).",
              "", "## Metric glossary", ""] + GLOSSARY + [""]
    wk = out / "week9"
    wk.mkdir(parents=True, exist_ok=True)
    for path, n, _, _ in rows:
        shutil.copyfile(out / path, wk / f"{n}.png")
    for old in wk.glob("*.csv"):
        old.unlink()
    (wk / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    block = ["<!-- week9:start -->", "## Week-9 figures (main set for the meeting and the report)", "",
             "Built by `python scripts/08_week9_figures.py` (D110); copies and the names / glossary tables in `week9/README.md`.", "",
             "| Chart | Takeaway | Source |", "|---|---|---|"] + [f"| `{p}` | {tk} | {src} |" for p, _, tk, src in sorted(rows)] + ["<!-- week9:end -->"]
    readme = out / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Final charts\n"
    if "<!-- week9:start -->" in text:
        a, rest = text.split("<!-- week9:start -->", 1)
        text = a + "\n".join(block) + rest.split("<!-- week9:end -->", 1)[1]
    else:
        head, sep, tail = text.partition("\n\n| Chart |")
        text = head + "\n\n" + "\n".join(block) + "\n\n## Other final charts" + sep + tail if sep else text + "\n" + "\n".join(block) + "\n"
    readme.write_text(text, encoding="utf-8", newline="\n")


def main(out=FIGURES):
    CREATED.clear()
    m, g = load()
    metrics_table(out, m, g)
    gain_table(out, m, g)
    s = std_models(m)
    titles = {"auc": ("03_auc", "Ranking quality (AUC)", True), "mrr": ("04_mrr", "How high the clicks appear (MRR)", False),
              "ndcg@5": ("05_ndcg5", "Quality of the top 5 (nDCG@5)", False), "ndcg@10": ("06_ndcg10", "Quality of the top 10 (nDCG@10)", False),
              "p@5": ("07_precision5", "Share of the top 5 clicked (Precision@5)", False),
              "r@10": ("08_recall10", "Share of clicks in the top 10 (Recall@10)", False)}
    for k, (name, phrase, ref) in titles.items():
        best = s.loc[s[k].idxmax(), "display_name"]
        metric_chart(out, m, g, k, name, f"{phrase}: best is {best}", takeaway_for(m, g, k), (0.5, "random") if ref else None)
    hit_rate(out, m, g)
    two_settings(out, m, g)
    gain_auc(out, m, g)
    metrics_explained(out)
    overview(out)
    readmes(out)
    return CREATED


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FIGURES))
    main(type(FIGURES)(ap.parse_args().out))
