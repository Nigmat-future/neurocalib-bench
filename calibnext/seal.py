"""One sealed two-animal validation and fixed numerical reproduction."""

from pathlib import Path
from statistics import fmean
from typing import Final, assert_never

from calib15.io import download_recording, read_study
from calib15.records import AnimalOutcome, Outcome, mean_r2, negative_days, validate_rows
from calib15.schema import Calibration, Row
from calibnext.evaluation import evaluate, evaluate_reference
from calibnext.records import read_frozen, require_final_access, study_dir
from calibnext.schema import Config, Frozen
from neurocalib.ledger import (
    Record,
    ResearchError,
    Started,
    environment,
    exclusive_run,
    file_digest,
    write_record,
)

PRIMARY_BUDGET: Final = 15
GATE_DELTA: Final = 0.02


class Run(Record):
    """A fixed comparison evaluated after the persistent final access marker."""

    label: str
    config: Config | Calibration
    frozen_sha256: str
    rows: tuple[Row, ...]


class Report(Record):
    """Both-animal final outcome, without test-time configuration selection."""

    frozen_sha256: str
    environment: dict[str, str]
    runs: tuple[Run, ...]
    outcome: Outcome
    matches_original: bool | None = None


def comparisons(frozen: Frozen) -> tuple[tuple[str, Config | Calibration], ...]:
    """Use the selected candidate and exactly the inherited three references."""
    return (
        ("winner", frozen.winner.config),
        *zip(
            ("baseline", "target_best", "frozen"),
            frozen.references,
            strict=True,
        ),
    )


def outcome(runs: tuple[Run, ...]) -> Outcome:
    """Keep the original per-animal harm and macro gain requirements."""
    winner = next(run for run in runs if run.label == "winner")
    baseline = next(run for run in runs if run.label == "baseline")
    summaries: list[AnimalOutcome] = []
    for animal in ("indy", "loco"):
        chosen = tuple(
            row
            for row in winner.rows
            if row.animal == animal and row.budget_seconds == PRIMARY_BUDGET
        )
        reference = tuple(
            row
            for row in baseline.rows
            if row.animal == animal and row.budget_seconds == PRIMARY_BUDGET
        )
        summaries.append(
            AnimalOutcome(
                animal=animal,
                delta=mean_r2(chosen) - mean_r2(reference),
                winner_negative_days=negative_days(chosen),
                baseline_negative_days=negative_days(reference),
            )
        )
    macro = fmean(item.delta for item in summaries)
    return Outcome(
        passed=macro + 1e-12 >= GATE_DELTA
        and all(
            item.delta + 1e-12 >= 0 and item.winner_negative_days <= item.baseline_negative_days
            for item in summaries
        ),
        macro_delta=macro,
        animals=tuple(summaries),
    )


def _run(root: Path, label: str, config: Config | Calibration, seal_hash: str) -> Run:
    match config:
        case Config():
            rows = evaluate(root, config, "test", (0, 5, 15, 30))
        case Calibration():
            rows = evaluate_reference(root, config, "test", (0, 5, 15, 30))
        case unreachable:
            assert_never(unreachable)
    return Run(
        label=label,
        config=config,
        frozen_sha256=seal_hash,
        rows=validate_rows(root, rows, "test", (0, 5, 15, 30)),
    )


def final(root: Path) -> Path:
    """Authorize downloads and validation only after verifying a passing seal."""
    with exclusive_run(study_dir(root)) as results:
        frozen = read_frozen(root)
        if frozen.environment != environment():
            message = "Runtime changed before final validation"
            raise ResearchError(message)
        seal_hash = file_digest(results / "frozen.json")
        marker = results / "final.started.json"
        if not marker.exists():
            write_record(marker, Started(frozen_sha256=seal_hash), exclusive=True)
        require_final_access(root)
        for spec in read_study(root).files:
            if spec.role == "test" or spec.animal == "loco":
                _ = download_recording(root, spec)
        runs: list[Run] = []
        for label, config in comparisons(frozen):
            path = results / "final" / f"{label}.json"
            if path.exists():
                run = Run.model_validate_json(path.read_bytes())
                if (run.label, run.config, run.frozen_sha256) != (label, config, seal_hash):
                    message = "Cached final result differs from the sealed comparison"
                    raise ResearchError(message)
                _ = validate_rows(root, run.rows, "test", (0, 5, 15, 30))
            else:
                run = _run(root, label, config, seal_hash)
                require_final_access(root)
                write_record(path, run)
            runs.append(run)
        output = results / "final" / "report.json"
        require_final_access(root)
        write_record(
            output,
            Report(
                frozen_sha256=seal_hash,
                environment=environment(),
                runs=tuple(runs),
                outcome=outcome(tuple(runs)),
            ),
        )
        return output


def same_rows(before: tuple[Row, ...], after: tuple[Row, ...]) -> bool:
    """Compare scientific fields while excluding measured calibration runtime."""
    import math  # noqa: PLC0415

    if len(before) != len(after):
        return False
    for left, right in zip(before, after, strict=True):
        if (left.animal, left.session, left.budget_seconds) != (
            right.animal,
            right.session,
            right.budget_seconds,
        ):
            return False
        fields = left.model_dump().keys() - {"animal", "session", "calibration_seconds"}
        if any(
            not math.isclose(
                getattr(left, field), getattr(right, field), abs_tol=1e-7, rel_tol=1e-7
            )
            for field in fields
        ):
            return False
    return True


def reproduce(root: Path) -> Path:
    """Repeat precisely the frozen final comparison, without new experiments."""
    with exclusive_run(study_dir(root)) as results:
        frozen = read_frozen(root)
        require_final_access(root)
        original = Report.model_validate_json((results / "final" / "report.json").read_bytes())
        seal_hash = file_digest(results / "frozen.json")
        if original.frozen_sha256 != seal_hash or tuple(
            (item.label, item.config) for item in original.runs
        ) != comparisons(frozen):
            message = "Reproduction requires the complete sealed comparison"
            raise ResearchError(message)
        runtime = environment()
        if any(runtime[name] != frozen.environment[name] for name in ("numpy", "h5py", "pydantic")):
            message = "Reproduction requires the locked scientific dependencies"
            raise ResearchError(message)
        runs = tuple(_run(root, label, config, seal_hash) for label, config in comparisons(frozen))
        matches = all(
            same_rows(old.rows, new.rows) for old, new in zip(original.runs, runs, strict=True)
        )
        output = results / "reproduced" / "report.json"
        require_final_access(root)
        write_record(
            output,
            Report(
                frozen_sha256=seal_hash,
                environment=runtime,
                runs=runs,
                outcome=outcome(runs),
                matches_original=matches,
            ),
        )
        if not matches:
            message = "Final numerical reproduction differs; evidence retained"
            raise ResearchError(message)
        return output
