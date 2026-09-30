"""Figures.

Four plots, each answering something the summary table cannot:

1. Where every cancer type lands, with the clinical baseline and the random
   control drawn as lines rather than rows, because they are references.
2. The margin over clinical-only per cancer type, which is the question the
   project exists to answer: does sequencing buy anything?
3. The whole grid, model against cancer type, because pooled means hide the
   cancer types where a model collapses.
4. Harrell's C against Uno's C, because when the two disagree the ranking
   depends on the censoring distribution rather than on the model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluate import margin_over
from .models import BASELINE, CONTROL

INK = "#2C2C2A"
MUTED = "#73726C"
HAIRLINE = "#D3D1C7"
ACCENT = "#1D9E75"
WARN = "#D85A30"
BASE_C = "#378ADD"
# Blue and orange are reserved for the baseline and the control, so the other
# models are drawn from a palette that never reuses them.
PALETTE = ["#1D9E75", "#7F77DD", "#BA7517", "#D4537E", "#2B8C8C", "#9B5DE5"]


def _colour_map(models) -> dict:
    """Stable colour per model, with the two references fixed."""
    others = [m for m in sorted(models) if m not in (BASELINE, CONTROL)]
    mapping = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(others)}
    mapping[BASELINE] = BASE_C
    mapping[CONTROL] = WARN
    return mapping


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HAIRLINE)
    ax.tick_params(colors=MUTED, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK)
    ax.set_axisbelow(True)


def _per_cancer(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    return results.groupby(["model", "cancer_type"])[metric].mean().reset_index()


def plot_c_index(results: pd.DataFrame, out: Path, dataset: str,
                 metric: str = "harrell_c") -> Path:
    """Headline: one point per cancer type, references drawn as vertical lines."""
    per_cancer = _per_cancer(results, metric)
    order = list(per_cancer.groupby("model")[metric].mean().sort_values().index)

    fig, ax = plt.subplots(figsize=(9, 0.85 * len(order) + 2.6))
    rng = np.random.default_rng(0)
    colours = _colour_map(per_cancer.model.unique())

    for row, (tag, colour, label) in enumerate((
        (BASELINE, BASE_C, "clinical only"),
        (CONTROL, WARN, "random signature"),
    )):
        if tag in per_cancer.model.values:
            value = per_cancer.loc[per_cancer.model == tag, metric].mean()
            ax.axvline(value, color=colour, lw=1.2, ls="--", zorder=1)
            # Stacked on separate lines: the two references often sit close
            # together, which is itself the point, so the labels must not.
            ax.text(value, -0.72 - 0.34 * row, label, color=colour, fontsize=9,
                    ha="center", va="center")

    for i, name in enumerate(order):
        values = per_cancer.loc[per_cancer.model == name, metric]
        colour = colours[name]
        ax.scatter(values, i + rng.uniform(-0.14, 0.14, len(values)),
                   s=30, color=colour, alpha=0.5, zorder=3, lw=0)
        ax.errorbar(values.mean(), i, xerr=values.std(), fmt="o", ms=9,
                    color=colour, elinewidth=1.4, capsize=4, zorder=4)
        ax.text(values.mean(), i + 0.32, f"{values.mean():.3f}",
                ha="center", fontsize=9, color=INK)

    ax.axvline(0.5, color=HAIRLINE, lw=1.0, zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    # Adapt to the data but always show 0.5, since that is the reference every
    # concordance index is read against.
    lo = min(0.48, float(per_cancer[metric].min()) - 0.03)
    hi = max(0.66, float(per_cancer[metric].max()) + 0.04)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1.45, len(order) - 0.3)
    ax.set_xlabel(f"{'Harrell' if metric == 'harrell_c' else 'Uno'} concordance index")
    ax.set_title(
        f"Prognostic accuracy by model — {dataset}\n"
        "one point per cancer type; 0.5 is random ordering",
        loc="left", fontsize=11, color=INK,
    )
    ax.grid(axis="x", color=HAIRLINE, lw=0.6, alpha=0.7)
    _style(ax)

    fig.tight_layout()
    path = out / f"{dataset}_c_index.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_margin_over_clinical(results: pd.DataFrame, out: Path, dataset: str,
                              metric: str = "harrell_c") -> Path:
    """Does sequencing buy anything? One point per cancer type, zero is the answer."""
    if BASELINE not in results.model.values:
        return None
    joined = margin_over(results, BASELINE, metric)
    per_cancer = joined.groupby(["model", "cancer_type"])["margin"].mean().reset_index()
    order = list(per_cancer.groupby("model")["margin"].mean().sort_values().index)

    fig, ax = plt.subplots(figsize=(9, 0.85 * len(order) + 2.4))
    rng = np.random.default_rng(1)
    ax.axvline(0, color=BASE_C, lw=1.3, ls="--", zorder=1)

    for i, name in enumerate(order):
        values = per_cancer.loc[per_cancer.model == name, "margin"]
        colour = ACCENT if values.mean() > 0 else WARN
        if name == CONTROL:
            colour = WARN
        ax.scatter(values, i + rng.uniform(-0.14, 0.14, len(values)),
                   s=30, color=colour, alpha=0.5, zorder=3, lw=0)
        ax.errorbar(values.mean(), i, xerr=values.std(), fmt="o", ms=9,
                    color=colour, elinewidth=1.4, capsize=4, zorder=4)
        won = int((values > 0).sum())
        ax.text(values.mean(), i + 0.32,
                f"{values.mean():+.3f}   wins {won}/{len(values)}",
                ha="center", fontsize=9, color=INK)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("concordance minus the clinical-only model, same cancer type")
    ax.set_title(
        f"What expression adds over age, sex and stage — {dataset}\n"
        "left of the dashed line means the sequencing made prognosis worse",
        loc="left", fontsize=11, color=INK,
    )
    ax.grid(axis="x", color=HAIRLINE, lw=0.6, alpha=0.7)
    _style(ax)

    fig.tight_layout()
    path = out / f"{dataset}_margin_over_clinical.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_grid(results: pd.DataFrame, out: Path, dataset: str,
              metric: str = "harrell_c") -> Path:
    """Model against cancer type. Pooled means hide collapses; this does not."""
    grid = (
        _per_cancer(results, metric)
        .pivot(index="model", columns="cancer_type", values=metric)
    )
    order = list(grid.mean(axis=1).sort_values().index)
    grid = grid.reindex(order)

    fig, ax = plt.subplots(
        figsize=(0.95 * grid.shape[1] + 4.0, 0.62 * len(grid) + 2.8)
    )
    im = ax.imshow(grid.to_numpy(), cmap="RdYlBu", vmin=0.40, vmax=0.80, aspect="auto")

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.to_numpy()[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.2f}".lstrip("0"), ha="center", va="center",
                    fontsize=8.5, color=INK)

    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(grid.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(grid.index, fontsize=9)
    ax.set_title(
        f"Concordance by cancer type — {dataset}\n"
        "blue is better; anything near 0.50 is no better than guessing",
        loc="left", fontsize=11, color=INK,
    )
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02).outline.set_visible(False)

    fig.tight_layout()
    path = out / f"{dataset}_grid.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_harrell_vs_uno(results: pd.DataFrame, out: Path, dataset: str) -> Path:
    """The two concordance estimators against each other.

    Points off the diagonal are cancer types where the ranking depends on the
    censoring distribution rather than on the model. That is worth seeing
    before quoting a single C-index to three decimal places.
    """
    valid = results.dropna(subset=["harrell_c", "uno_c"])
    if valid.empty:
        return None
    per_cancer = (
        valid.groupby(["model", "cancer_type"])[["harrell_c", "uno_c"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    lo = float(min(per_cancer.harrell_c.min(), per_cancer.uno_c.min())) - 0.03
    hi = float(max(per_cancer.harrell_c.max(), per_cancer.uno_c.max())) + 0.03
    ax.plot([lo, hi], [lo, hi], color=HAIRLINE, lw=1.2, zorder=1)

    colours = _colour_map(per_cancer.model.unique())
    for name, sub in per_cancer.groupby("model"):
        ax.scatter(sub.harrell_c, sub.uno_c, s=42, color=colours[name], alpha=0.75,
                   lw=0, label=name, zorder=3)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Harrell's C (depends on the test set's censoring)")
    ax.set_ylabel("Uno's C (inverse-probability-of-censoring weighted)")
    ax.set_title(
        f"Do the two concordance estimators agree? — {dataset}\n"
        "points off the line are rankings that depend on follow-up, not on the model",
        loc="left", fontsize=11, color=INK,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    ax.grid(color=HAIRLINE, lw=0.6, alpha=0.6)
    _style(ax)

    fig.tight_layout()
    path = out / f"{dataset}_harrell_vs_uno.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def make_all(results: pd.DataFrame, out_dir: Path, dataset: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_c_index(results, out_dir, dataset),
        plot_margin_over_clinical(results, out_dir, dataset),
        plot_grid(results, out_dir, dataset),
        plot_harrell_vs_uno(results, out_dir, dataset),
    ]
    return [p for p in paths if p is not None]
