"""Charts for the week-7 behaviour experiments, in the style of scripts/05_presentation_charts.py: white background, light gridlines,
Okabe-Ito colours (no red or green), large fonts, a plain-language title and a one-line caption (what it shows, what to take away).

Inputs: small tables of numbers already saved as CSV by the experiment scripts. Outputs: PNGs at 200 dpi.
"""
import textwrap
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREY, SKY, PINK, YELLOW, VERMILLION, PURPLE = "#0072B2", "#E69F00", "#999999", "#56B4E9", "#CC79A7", "#F0E442", "#D55E00", "#6A3D9A"
PALETTE = [BLUE, ORANGE, SKY, PINK, GREY, YELLOW, PURPLE, VERMILLION]
INK = "#1a1a1a"
DPI = 200


def _style():
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "axes.titlesize": 19, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "axes.edgecolor": GREY, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def new(title, grid="y", size=(12, 6.75)):
    """A figure with one axis, the title, light gridlines on the value axis, no top or right frame."""
    _style()
    fig, ax = plt.subplots(figsize=size, facecolor="white")
    ax.set_title(title, pad=14)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#e6e6e6")
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.2)
    return fig, ax


def save(fig, path, caption):
    """Add the one-line caption under the chart and write the PNG (retried: an open preview can lock the file on Windows)."""
    fig.text(0.012, 0.02, "\n".join(textwrap.wrap(caption, 125)), fontsize=12.5, color="#444444", ha="left", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            fig.savefig(path, dpi=DPI, facecolor="white")
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2)
    plt.close(fig)
    return path


def _err(values, low, high):
    if low is None:
        return None
    v = np.asarray(values, dtype=float)
    return [np.clip(v - np.asarray(low, dtype=float), 0, None), np.clip(np.asarray(high, dtype=float) - v, 0, None)]


def bars(path, title, caption, labels, values, ylabel, low=None, high=None, colors=None, fmt="{:+.4f}", zero=True, hline=None,
         hline_label=None, notes=None, horizontal=False, xlabel=None, base=0.0, lim=None):
    """Simple bars with value labels and optional CI error bars, zero line and reference line; notes = extra text under each label.
    base: where bars start (e.g. 0.5 for AUC); lim: value-axis limits."""
    fig, ax = new(title, grid="x" if horizontal else "y")
    n = len(values)
    colors = colors or [BLUE] * n
    err = _err(values, low, high)
    pos = np.arange(n)
    if horizontal:
        ax.barh(pos, np.asarray(values, dtype=float) - base, left=base, color=colors, height=0.62, xerr=err, error_kw={"elinewidth": 2, "capsize": 5, "ecolor": INK})
        ax.set_yticks(pos, labels)
        ax.set_xlabel(ylabel)
        span = np.nanmax(np.abs(np.r_[values, high if high is not None else values])) or 1
        for y, v, h in zip(pos, values, high if high is not None else values):
            ax.text((h if v >= 0 else min(v, h)) + 0.02 * span * (1 if v >= 0 else -1), y, fmt.format(v), va="center",
                    ha="left" if v >= 0 else "right", fontsize=13, fontweight="bold")
        if zero:
            ax.axvline(0, color=INK, linewidth=1.5)
        if lim:
            ax.set_xlim(*lim)
    else:
        ax.bar(pos, np.asarray(values, dtype=float) - base, bottom=base, color=colors, width=0.62, yerr=err, error_kw={"elinewidth": 2, "capsize": 6, "ecolor": INK})
        ax.set_xticks(pos, [f"{l}\n{notes[i]}" if notes else l for i, l in enumerate(labels)])
        ax.set_ylabel(ylabel)
        top = np.asarray(high if high is not None else values, dtype=float)
        bottom = np.asarray(low if low is not None else values, dtype=float)
        span = float(np.nanmax(np.abs(np.r_[top, bottom] - base))) or 1.0
        for x, v, t, b in zip(pos, values, top, bottom):
            up = v >= 0
            ax.text(x, (t if up else b) + (0.03 if up else -0.03) * span, fmt.format(v), ha="center", va="bottom" if up else "top",
                    fontsize=13, fontweight="bold")
        if zero:
            ax.axhline(0, color=INK, linewidth=1.5)
        lo_lim = min(base, float(np.nanmin(bottom))) if zero else float(np.nanmin(bottom))
        ax.set_ylim(*(lim or (lo_lim - 0.18 * span, float(np.nanmax(top)) + 0.2 * span)))
    if xlabel and not horizontal:
        ax.set_xlabel(xlabel)
    if hline is not None:
        (ax.axvline if horizontal else ax.axhline)(hline, color=GREY, linestyle="--", linewidth=1.8)
        if hline_label:
            if horizontal:
                ax.text(hline, n - 0.4, f" {hline_label}", color=INK, fontsize=12, va="bottom")
            else:
                ax.text(n - 0.5, hline, hline_label, color=INK, fontsize=12, ha="right", va="bottom")
    return save(fig, path, caption)


def grouped(path, title, caption, groups, series, ylabel, fmt="{:.3f}", zero=False, ylim=None, notes=None, legend_loc="best"):
    """Grouped bars. series: [(name, values, low or None, high or None, colour)]."""
    fig, ax = new(title)
    k = len(series)
    width = 0.8 / k
    pos = np.arange(len(groups))
    allv = []
    for j, (name, values, low, high, color) in enumerate(series):
        x = pos + (j - (k - 1) / 2) * width
        ax.bar(x, values, width=width * 0.95, color=color, label=name, yerr=_err(values, low, high),
               error_kw={"elinewidth": 1.8, "capsize": 4, "ecolor": INK})
        top = np.asarray(high if high is not None else values, dtype=float)
        for xi, v, t in zip(x, values, top):
            if np.isfinite(v):
                ax.text(xi, t if v >= 0 else min(v, t), fmt.format(v), ha="center", va="bottom", fontsize=11.5, fontweight="bold")
        allv += list(np.r_[values, top])
    ax.set_xticks(pos, [f"{g}\n{notes[i]}" if notes else g for i, g in enumerate(groups)])
    ax.set_ylabel(ylabel)
    if zero:
        ax.axhline(0, color=INK, linewidth=1.5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=13, loc=legend_loc)
    return save(fig, path, caption)


def lines(path, title, caption, x, series, xlabel, ylabel, fmt="{:.4f}", zero=False, xticks=None, hline=None, hline_label=None):
    """Lines with markers and value labels. series: [(name, values, low or None, high or None, colour, marker)]."""
    fig, ax = new(title, grid="both")
    for name, values, low, high, color, marker in series:
        ax.plot(x, values, color=color, marker=marker, markersize=8, linewidth=2.4, label=name)
        if low is not None:
            ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
        for xi, v in zip(x, values):
            ax.annotate(fmt.format(v), (xi, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=11, color=INK)
    if zero:
        ax.axhline(0, color=INK, linewidth=1.5)
    if hline is not None:
        ax.axhline(hline, color=GREY, linestyle="--", linewidth=1.8)
        if hline_label:
            ax.text(x[-1], hline, hline_label, color=INK, fontsize=12, ha="right", va="bottom")
    if xticks is not None:
        ax.set_xticks(x, xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=13)
    return save(fig, path, caption)


def histogram(path, title, caption, values, xlabel, cuts=None, cut_labels=None, bins=50, color=BLUE):
    """Histogram with optional vertical cut-off lines."""
    fig, ax = new(title)
    v = np.asarray(values, dtype=float)
    ax.hist(v[np.isfinite(v)], bins=bins, color=color, edgecolor="white", linewidth=0.5)
    for i, c in enumerate(cuts or []):
        ax.axvline(c, color=INK, linestyle="--", linewidth=1.8)
        if cut_labels:
            ax.text(c, ax.get_ylim()[1] * 0.97, f" {cut_labels[i]}", fontsize=12, va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    return save(fig, path, caption)


def heatmap(path, title, caption, matrix, rows, cols, xlabel, ylabel, fmt="{:.2f}", center=None):
    """Heatmap with a value in every cell; a diverging blue-orange scale when `center` is given."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    fig, ax = new(title, grid=None)
    m = np.asarray(matrix, dtype=float)
    if center is not None:
        cmap = LinearSegmentedColormap.from_list("bo", [BLUE, "#ffffff", ORANGE])
        lim = float(np.nanmax(np.abs(m - center))) or 1.0
        im = ax.imshow(m, cmap=cmap, norm=TwoSlopeNorm(center, center - lim, center + lim), aspect="auto")
    else:
        im = ax.imshow(m, cmap=LinearSegmentedColormap.from_list("b", ["#ffffff", BLUE]), aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=10.5, color=INK)
    ax.set_xticks(range(len(cols)), cols, rotation=0)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    return save(fig, path, caption)


def scatter_map(path, title, caption, x, y, labels, names, xlabel, ylabel, sample=4000, seed=42):
    """2-D map of points coloured by group (a seeded sample) with the group means marked and named."""
    fig, ax = new(title, grid="both")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(sample, len(x)), replace=False)
    labels = np.asarray(labels)
    for k, name in enumerate(names):
        sel = idx[labels[idx] == k]
        ax.scatter(np.asarray(x)[sel], np.asarray(y)[sel], s=9, alpha=0.45, color=PALETTE[k % len(PALETTE)], label=name)
        mx, my = np.asarray(x)[labels == k].mean(), np.asarray(y)[labels == k].mean()
        ax.scatter([mx], [my], s=260, marker="X", color=PALETTE[k % len(PALETTE)], edgecolor=INK, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=12, markerscale=2.5, loc="best")
    return save(fig, path, caption)


def contact_sheet(paths, out, title, cols=3):
    """One page with several saved charts, for a quick overview."""
    import matplotlib.image as mpimg
    _style()
    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4.6 * rows + 0.8), facecolor="white")
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, p in zip(np.atleast_1d(axes).ravel(), paths):
        ax.imshow(mpimg.imread(str(p)))
    fig.suptitle(title, fontsize=24, fontweight="bold", x=0.01, ha="left")
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.8 / (4.6 * rows + 0.8), bottom=0.005, wspace=0.02, hspace=0.04)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    return out
