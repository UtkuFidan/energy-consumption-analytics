"""Small, consistent figures for the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERMILION, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#6b6b6b"

plt.rcParams.update(
    {
        "figure.dpi": 110, "savefig.dpi": 110, "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    }
)  # fmt: skip


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def forecast_vs_actual(pred: pd.DataFrame, start: str, end: str, path: Path, lines: dict,
                       title: str, ylabel: str, tz: str | None = None) -> None:  # fmt: skip
    """Actual load, conformalised interval and forecasts; ``lines`` maps col to (colour, label)."""
    lo, hi = pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz)
    p = pred[(pred.index >= lo) & (pred.index < hi)]
    x = p.index.tz_convert(tz) if tz else p.index
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.fill_between(x, p["q10_cp"], p["q90_cp"], color=BLUE, alpha=0.15, lw=0,
                    label="80% interval, conformalised")  # fmt: skip
    ax.plot(x, p["actual"], color="#222222", lw=1.3, label="Actual")
    for col, (colour, label) in lines.items():
        ax.plot(x, p[col], color=colour, lw=1.0, label=label)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, path)


def error_by_hour(table: pd.DataFrame, path: Path, styles: dict, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for col, (c, lab) in styles.items():
        ax.plot(table.index, table[col], color=c, lw=1.6, marker="o", ms=3, label=lab)
    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(0, 24, 3))
    ax.set_title(title, loc="left")
    ax.legend()
    _save(fig, path)


def mae_by_month(table: pd.DataFrame, path: Path, styles: dict, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    for col, (c, lab) in styles.items():
        ax.plot(table.index, table[col], color=c, lw=1.6, marker="o", ms=3, label=lab)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(ncol=len(styles))
    _save(fig, path)


def archetypes(centroids: pd.DataFrame, shares: pd.Series, path: Path) -> None:
    k = len(centroids)
    cols = min(k, 3)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.0 * cols, 2.2 * rows), sharey=True, squeeze=False
    )
    h = np.arange(24)
    for a, ax in enumerate(axes.flat):
        if a >= k:
            ax.axis("off")
            continue
        wd = centroids.loc[a, [c for c in centroids if c.startswith("wd_")]].to_numpy()
        we = centroids.loc[a, [c for c in centroids if c.startswith("we_")]].to_numpy()
        ax.plot(h, wd, color=BLUE, lw=1.6, label="Weekday")
        ax.plot(h, we, color=ORANGE, lw=1.6, label="Weekend")
        ax.axhline(1, color=GREY, lw=0.6)
        ax.set_title(f"Archetype {a} ({shares[a]:.0f}% of clients)", loc="left", fontsize=9)
        ax.set_xticks([0, 6, 12, 18, 23])
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("Hour of day", fontsize=9)
    fig.supylabel("Load / client mean", fontsize=9)
    _save(fig, path)


def pr_curve(y: pd.Series, scores: dict[str, pd.Series], budget_k: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    colors = {"xgboost": BLUE, "isolation_forest": ORANGE}
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=colors.get(name, GREEN), lw=1.8, label=name.replace("_", " "))
    ax.axhline(y.mean(), color=GREY, ls="--", lw=1, label=f"Base rate {y.mean():.3f}")
    top = scores["xgboost"].sort_values(ascending=False).head(budget_k).index
    ax.scatter([y.loc[top].sum() / y.sum()], [y.loc[top].mean()], color=VERMILION, zorder=5,
               label=f"Inspection budget (top {budget_k})")  # fmt: skip
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "Theft detection, precision-recall (SGCC, 5-fold out-of-fold)", loc="left", fontsize=9
    )
    ax.legend(fontsize=8)
    _save(fig, path)


def k_selection(sel: pd.DataFrame, chosen: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8))
    panels = [
        ("silhouette", "Silhouette (higher is better)", BLUE),
        ("inertia", "Within-cluster sum of squares (elbow)", ORANGE),
        ("stability_ari", "Subsample stability, ARI", GREEN),
    ]
    for ax, (col, title, c) in zip(axes, panels, strict=True):
        ax.plot(sel["k"], sel[col], color=c, lw=1.6, marker="o", ms=4)
        bad = ~sel["eligible"]
        ax.scatter(sel.loc[bad, "k"], sel.loc[bad, col], facecolors="white", edgecolors=c, zorder=5)
        ax.axvline(chosen, color=GREY, ls="--", lw=0.8)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("k")
    axes[0].text(0.02, 0.05, "open marker: smallest cluster below 2% of clients",
                 transform=axes[0].transAxes, fontsize=7, color=GREY)  # fmt: skip
    _save(fig, path)
