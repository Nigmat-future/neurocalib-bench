"""CSV exports and static figures from saved evidence, without recording access."""

# Matplotlib's extensible keyword APIs expose incomplete library typing.
# pyright: reportUnknownMemberType=false

import csv
from pathlib import Path
from statistics import fmean

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from calib15.io import results_dir
from calib15.records import Gate, Report, Trial, negative_days, read_events
from calib15.schema import Row


def _pending(axis: Axes, message: str) -> None:
    """Make missing evidence explicit instead of implying an observed outcome."""
    _ = axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()


def _baseline_plot(axis: Axes, trials: tuple[Trial, ...]) -> None:
    """Show every completed baseline strength and the observed penalty boundaries."""
    baselines = tuple(event for event in trials if event.kind == "baseline")
    if not baselines:
        _pending(axis, "Baseline evaluation pending")
        return
    for mode in ("target", "residual"):
        strengths = sorted(
            {event.config.strength for event in baselines if event.config.mode == mode}
        )
        for strength in strengths:
            group = sorted(
                (
                    event
                    for event in baselines
                    if event.config.mode == mode and event.config.strength == strength
                ),
                key=lambda event: event.config.alpha,
            )
            _ = axis.plot(
                [event.config.alpha for event in group],
                [fmean(row.r2 for row in event.rows) for event in group],
                marker="o",
                label=f"{mode}, strength={strength:g}",
            )
    penalties = [event.config.alpha for event in baselines]
    for boundary in sorted({min(penalties), max(penalties)}):
        _ = axis.axvline(boundary, color="0.75", linestyle=":", linewidth=0.8)
    axis.set_xscale("log")
    _ = axis.set(
        xlabel="Ridge penalty alpha (log scale)",
        ylabel="Mean development velocity R²",
        title="Completed baseline sweep; dotted observed limits",
    )
    _ = axis.axhline(0, color="0.75", linewidth=0.7)
    _ = axis.legend(fontsize=8, frameon=False)


def _paired_plot(axis: Axes, trials: tuple[Trial, ...], output: Path) -> None:
    """Compare saved daily results descriptively without recomputing a gate."""
    baselines = tuple(event for event in trials if event.kind == "baseline")
    candidates = tuple(event for event in trials if event.kind == "candidate")
    if not baselines or not candidates:
        _pending(axis, "Candidate comparison pending\nRequires completed baseline and candidate")
        return
    baseline = max(baselines, key=lambda event: fmean(row.r2 for row in event.rows))
    candidate = max(candidates, key=lambda event: fmean(row.r2 for row in event.rows))
    candidate_label = "Best completed candidate"
    decision_path = output / "decision.json"
    if decision_path.exists():
        decision = Gate.model_validate_json(decision_path.read_bytes())
        selected = next((event for event in candidates if event.config == decision.winner), None)
        if selected is not None:
            candidate, candidate_label = selected, "Recorded gate candidate"
        baseline = next(
            (event for event in baselines if event.config == decision.baseline_best), baseline
        )
    before = {(row.animal, row.session): row.r2 for row in baseline.rows}
    after = {(row.animal, row.session): row.r2 for row in candidate.rows}
    days = sorted(before.keys() & after.keys())
    if not days:
        _pending(axis, "Candidate comparison pending\nNo paired recorded days")
        return
    for event, values, label in (
        (baseline, before, "Best completed baseline"),
        (candidate, after, candidate_label),
    ):
        _ = axis.plot(
            range(len(days)),
            [values[day] for day in days],
            marker="o",
            label=f"{label}: {event.config.mode}",
        )
    _ = axis.set_xticks(
        range(len(days)),
        [day[1].split("_")[1] for day in days],
        rotation=45,
        ha="right",
    )
    _ = axis.set(
        xlabel="Development recording date", ylabel="Velocity R²", title="Paired saved scores"
    )
    _ = axis.axhline(0, color="0.75", linewidth=0.7)
    _ = axis.legend(fontsize=8, frameon=False)


def _final_plot(path: Path) -> None:
    """Plot animal-specific mean curves only when final evidence already exists."""
    report = Report.model_validate_json(path.read_bytes())
    animals = sorted({row.animal for run in report.runs for row in run.rows})
    if not animals:
        return
    figure = Figure(figsize=(6 * len(animals), 4.5), layout="constrained")
    for index, animal in enumerate(animals, start=1):
        axis = figure.add_subplot(1, len(animals), index)
        for run in report.runs:
            rows = tuple(row for row in run.rows if row.animal == animal)
            budgets = sorted({row.budget_seconds for row in rows})
            _ = axis.plot(
                budgets,
                [
                    fmean(row.r2 for row in rows if row.budget_seconds == budget)
                    for budget in budgets
                ],
                marker="o",
                label=run.label.replace("_", " "),
            )
        _ = axis.set(
            xlabel="Observed target prefix (s)",
            ylabel="Mean test-day velocity R²",
            title=f"{animal.capitalize()}: saved final comparison",
        )
        _ = axis.axhline(0, color="0.75", linewidth=0.7)
        _ = axis.legend(fontsize=8, frameon=False)
    figure.savefig(path.parent / "calibration_curves.png", dpi=180, facecolor="white")


def render(root: Path) -> Path:
    """Export every terminal trial and plot completed saved comparisons only."""
    output = results_dir(root)
    output.mkdir(parents=True, exist_ok=True)
    trials = tuple(
        event for event in read_events(output / "experiments.jsonl") if event.status != "running"
    )
    fields: list[str] = [
        "trial_id",
        "mode",
        "alpha",
        "strength",
        "components",
        "kind",
        "status",
        "score",
        "negative_days",
        "sequence",
    ]
    with (output / "search_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                "trial_id": event.trial_id,
                **event.config.model_dump(),
                "kind": event.kind,
                "status": event.status,
                "score": event.score,
                "negative_days": negative_days(event.rows) if event.rows else None,
                "sequence": event.sequence,
            }
            for event in trials
        )
    with (output / "development.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("trial_id", "mode", "alpha", "strength", "components", *Row.model_fields),
        )
        writer.writeheader()
        writer.writerows(
            {"trial_id": event.trial_id, **event.config.model_dump(), **row.model_dump()}
            for event in trials
            for row in event.rows
        )
    complete = tuple(event for event in trials if event.score is not None and event.rows)
    figure = Figure(figsize=(12, 4.8), layout="constrained")
    _baseline_plot(figure.add_subplot(121), complete)
    _paired_plot(figure.add_subplot(122), complete, output)
    _ = figure.suptitle("Fifteen-second calibration | Saved development evidence", fontsize=12)
    image = output / "calibration_comparison.png"
    figure.savefig(image, dpi=180, facecolor="white")
    final = output / "final/report.json"
    if final.exists():
        _final_plot(final)
    return image
