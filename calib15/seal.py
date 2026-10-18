"""Bind research inputs and guard every held-out evaluation behind one seal."""

import math
from pathlib import Path
from statistics import fmean
from typing import Final

from calib15.evaluation import evaluate
from calib15.io import read_study, study_dir
from calib15.records import (
    EPSILON,
    AnimalOutcome,
    Frozen,
    Outcome,
    Report,
    Run,
    mean_r2,
    negative_days,
    read_frozen,
    require_final_access,
    validate_rows,
)
from neurocalib.ledger import (
    ResearchError,
    Started,
    environment,
    exclusive_run,
    file_digest,
    write_record,
)

TOLERANCE: Final = 1e-7


def outcome(root: Path, frozen: Frozen, runs: tuple[Run, ...]) -> Outcome:
    """Require both animals to meet the predeclared gain and negative-transfer rules."""
    study = read_study(root)
    winner = next(run for run in runs if run.label == "winner")
    baseline = next(run for run in runs if run.config == frozen.decision.baseline_best)
    summaries: list[AnimalOutcome] = []
    for animal in sorted({item.animal for item in study.files if item.role == "test"}):
        candidate_rows = tuple(
            row
            for row in winner.rows
            if row.animal == animal and row.budget_seconds == study.primary_budget_seconds
        )
        baseline_rows = tuple(
            row
            for row in baseline.rows
            if row.animal == animal and row.budget_seconds == study.primary_budget_seconds
        )
        summaries.append(
            AnimalOutcome(
                animal=animal,
                delta=mean_r2(candidate_rows) - mean_r2(baseline_rows),
                winner_negative_days=negative_days(candidate_rows),
                baseline_negative_days=negative_days(baseline_rows),
            )
        )
    macro = fmean(item.delta for item in summaries)
    passed = macro + EPSILON >= study.gate_delta and all(
        item.delta + EPSILON >= 0 and item.winner_negative_days <= item.baseline_negative_days
        for item in summaries
    )
    return Outcome(passed=passed, macro_delta=macro, animals=tuple(summaries))


def final(root: Path) -> Path:
    """Open the final gate once; failures may resume only the same four methods."""
    with exclusive_run(study_dir(root)) as results:
        frozen = read_frozen(root)
        if frozen.environment != environment():
            message = "Runtime changed before final evaluation"
            raise ResearchError(message)
        marker = results / "final.started.json"
        frozen_hash = file_digest(results / "frozen.json")
        if not marker.exists():
            write_record(marker, Started(frozen_sha256=frozen_hash), exclusive=True)
        require_final_access(root)
        runs: list[Run] = []
        for comparison in frozen.comparisons:
            path = results / "final" / f"{comparison.label}.json"
            if path.exists():
                run = Run.model_validate_json(path.read_bytes())
                if (
                    run.config != comparison.config
                    or run.frozen_sha256 != frozen_hash
                    or run.label != comparison.label
                ):
                    message = "Completed final result differs from the frozen method"
                    raise ResearchError(message)
                _ = validate_rows(root, run.rows, "test", read_study(root).budgets_seconds)
            else:
                rows = validate_rows(
                    root,
                    evaluate(
                        root,
                        frozen.features,
                        comparison.config,
                        "test",
                        budgets=read_study(root).budgets_seconds,
                    ),
                    "test",
                    read_study(root).budgets_seconds,
                )
                require_final_access(root)
                run = Run(
                    label=comparison.label,
                    config=comparison.config,
                    frozen_sha256=frozen_hash,
                    rows=rows,
                )
                write_record(path, run)
            runs.append(run)
        output = results / "final" / "report.json"
        write_record(
            output,
            Report(
                frozen_sha256=frozen_hash,
                environment=environment(),
                runs=tuple(runs),
                outcome=outcome(root, frozen, tuple(runs)),
            ),
        )
        return output


def scientific_equal(original: Run, repeated: Run) -> bool:
    """Ignore elapsed machine time while comparing every scientific row field."""
    before = {(row.animal, row.session, row.budget_seconds): row for row in original.rows}
    after = {(row.animal, row.session, row.budget_seconds): row for row in repeated.rows}
    if before.keys() != after.keys() or len(before) != len(original.rows):
        return False
    for key, row in before.items():
        left, right = row.model_dump(), after[key].model_dump()
        for field in left.keys() - {"animal", "session", "calibration_seconds"}:
            if not math.isclose(left[field], right[field], rel_tol=TOLERANCE, abs_tol=TOLERANCE):
                return False
    return True


def reproduce(root: Path) -> Path:
    """Rerun precisely the completed final comparison without another selection step."""
    with exclusive_run(study_dir(root)) as results:
        frozen = read_frozen(root)
        require_final_access(root)
        original = Report.model_validate_json((results / "final" / "report.json").read_bytes())
        frozen_hash = file_digest(results / "frozen.json")
        if original.frozen_sha256 != frozen_hash or tuple(
            (run.label, run.config) for run in original.runs
        ) != tuple((item.label, item.config) for item in frozen.comparisons):
            message = "Reproduction requires the complete original frozen comparison"
            raise ResearchError(message)
        runtime = environment()
        if any(runtime[name] != frozen.environment[name] for name in ("numpy", "h5py", "pydantic")):
            message = "Reproduction requires the locked scientific dependency versions"
            raise ResearchError(message)
        runs = tuple(
            Run(
                label=item.label,
                config=item.config,
                frozen_sha256=frozen_hash,
                rows=validate_rows(
                    root,
                    evaluate(
                        root,
                        frozen.features,
                        item.config,
                        "test",
                        budgets=read_study(root).budgets_seconds,
                    ),
                    "test",
                    read_study(root).budgets_seconds,
                ),
            )
            for item in frozen.comparisons
        )
        require_final_access(root)
        matches = all(
            scientific_equal(left, right) for left, right in zip(original.runs, runs, strict=True)
        )
        output = results / "reproduced" / "report.json"
        write_record(
            output,
            Report(
                frozen_sha256=frozen_hash,
                environment=runtime,
                runs=runs,
                outcome=outcome(root, frozen, runs),
                matches_original=matches,
            ),
        )
        if not matches:
            message = f"Scientific reproduction differs; retained evidence at {output}"
            raise ResearchError(message)
        return output
