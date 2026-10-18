"""Standalone research figures from saved scores, with no recording access."""

# Matplotlib exposes untyped extensible keyword arguments on Figure methods.
# pyright: reportUnknownMemberType=false

from pathlib import Path
from statistics import fmean
from typing import Final

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from neurocalib.ledger import FinalReport

COLORS: Final = ("#136f63", "#26364a", "#b46f24", "#7a549e", "#ba5650", "#347aaa", "#7b7d86")


def render(root: Path) -> Path:
    """Render macro-day R² curves and paired per-day calibration outcomes."""
    path = root / "results" / "final" / "report.json"
    report = FinalReport.model_validate_json(path.read_bytes())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for index, result in enumerate(report.results):
        budgets = sorted({row.budget_seconds for row in result.rows})
        scores = [
            fmean(row.r2 for row in result.rows if row.budget_seconds == budget)
            for budget in budgets
        ]
        label = result.label.replace("baseline_", "").replace("_", " ")
        _ = axes[0].plot(
            budgets,
            scores,
            marker="o",
            markersize=4,
            linewidth=2 if result.label == "winner" else 1.2,
            color=COLORS[index % len(COLORS)],
            label=label,
        )
    axes[0].set(
        xlabel="Target-day prefix available (s)",
        ylabel="Mean velocity R² across test days",
        title="Calibration budget and held-out performance",
    )
    _ = axes[0].axhline(0, color="#b7bdc3", linewidth=0.7)
    _ = axes[0].legend(fontsize=8, frameon=False)
    winner = next(item for item in report.results if item.label == "winner")
    for index, name in enumerate(sorted({row.session for row in winner.rows})):
        rows = sorted(
            (row for row in winner.rows if row.session == name), key=lambda row: row.budget_seconds
        )
        _ = axes[1].plot(
            [row.budget_seconds for row in rows],
            [row.r2 for row in rows],
            marker="o",
            color=COLORS[index],
            label=name.removeprefix("indy_").removesuffix(".mat"),
        )
    axes[1].set(
        xlabel="Target-day prefix available (s)",
        ylabel="Velocity R²",
        title="Selected pipeline: each test day",
    )
    _ = axes[1].axhline(0, color="#b7bdc3", linewidth=0.7)
    _ = axes[1].legend(frameon=False)
    _ = figure.suptitle("Neurocalib Bench | One animal, two locked test days", fontsize=12)
    output = path.parent / "calibration_curves.png"
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)
    return output
