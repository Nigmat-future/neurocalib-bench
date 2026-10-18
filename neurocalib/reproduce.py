"""Recompute only a completed, frozen comparison on another machine."""

import math
from pathlib import Path
from typing import Final

from neurocalib.ledger import (
    FinalReport,
    FinalResult,
    Frozen,
    Metric,
    Record,
    ResearchError,
    Started,
    comparisons,
    environment,
    exclusive_run,
    file_digest,
    fingerprints,
    write_record,
)
from neurocalib.research import checked_rows

TOLERANCE: Final = 1e-7


class Reproduction(Record):
    """Actual rerun results and its environment, separate from the original final."""

    frozen_sha256: str
    original_report_sha256: str
    environment: dict[str, str]
    relative_tolerance: float = TOLERANCE
    absolute_tolerance: float = TOLERANCE
    matches: bool
    results: tuple[FinalResult, ...]


def scientific_values(row: Metric) -> tuple[float, ...]:
    """Compare scientific outputs while excluding measured machine timings."""
    return (
        row.r2,
        row.rmse,
        row.r2_x,
        row.r2_y,
        row.zero_r2,
        row.zero_rmse,
        row.n_scoring_bins,
        row.n_prefix_bins,
        row.n_channels,
        row.n_features,
        row.missing_channels,
        row.last_bin_seconds,
    )


def equivalent(original: FinalResult, rerun: FinalResult) -> bool:
    """Require identical scoring cells and numerically consistent scientific values."""
    expected = {(row.session, row.mode, row.budget_seconds): row for row in original.rows}
    observed = {(row.session, row.mode, row.budget_seconds): row for row in rerun.rows}
    if len(expected) != len(original.rows) or expected.keys() != observed.keys():
        return False
    return all(
        math.isclose(left, right, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
        for key, row in expected.items()
        for left, right in zip(
            scientific_values(row), scientific_values(observed[key]), strict=True
        )
    )


def reproduce(root: Path) -> Path:
    """Rerun fixed final strategies without permitting new selection or parameters."""
    with exclusive_run(root) as results:
        frozen_path = results / "frozen.json"
        original_path = results / "final" / "report.json"
        if not original_path.exists():
            message = "Reproduction requires a completed final evaluation"
            raise ResearchError(message)
        frozen = Frozen.model_validate_json(frozen_path.read_bytes())
        frozen_hash = file_digest(frozen_path)
        original = FinalReport.model_validate_json(original_path.read_bytes())
        marker = Started.model_validate_json((results / "final.started.json").read_bytes())
        if (
            frozen.inputs != fingerprints(root)
            or frozen.ledger_sha256 != file_digest(results / "experiments.jsonl")
            or original.frozen_sha256 != frozen_hash
            or marker.frozen_sha256 != frozen_hash
        ):
            message = "Code, protocol, dependencies, ledger or final lock changed"
            raise ResearchError(message)
        declared = comparisons(frozen)
        if tuple((report.label, report.strategy) for report in original.results) != declared:
            message = "Completed final report differs from the frozen comparison"
            raise ResearchError(message)
        runtime = environment()
        if any(
            runtime[name] != frozen.winner.environment[name]
            for name in ("numpy", "h5py", "pydantic")
        ):
            message = "Install the locked scientific dependency versions before reproduction"
            raise ResearchError(message)
        reruns = tuple(
            FinalResult(
                label=label,
                strategy=strategy,
                frozen_sha256=frozen_hash,
                rows=checked_rows(root, strategy, "test"),
            )
            for label, strategy in declared
        )
        if frozen.inputs != fingerprints(root):
            message = "Immutable inputs changed during reproduction"
            raise ResearchError(message)
        matches = all(
            equivalent(before, after)
            for before, after in zip(original.results, reruns, strict=True)
        )
        output = results / "reproduced" / "report.json"
        write_record(
            output,
            Reproduction(
                frozen_sha256=frozen_hash,
                original_report_sha256=file_digest(original_path),
                environment=runtime,
                matches=matches,
                results=reruns,
            ),
        )
        if not matches:
            message = f"Scientific reproduction differs; inspect {output}"
            raise ResearchError(message)
        return output
