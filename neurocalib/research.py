"""Budgeted development experiments and a sealed final evaluation."""

import csv
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Literal

from neurocalib.evaluation import evaluate
from neurocalib.ledger import (
    BASELINES,
    MAX_TRIALS,
    FinalReport,
    FinalResult,
    Frozen,
    Metric,
    ResearchError,
    Started,
    Trial,
    append_trial,
    comparisons,
    digest,
    environment,
    exclusive_run,
    file_digest,
    fingerprints,
    read_trials,
    write_record,
)
from neurocalib.schema import Protocol, Strategy


def checked_rows(
    root: Path, strategy: Strategy, phase: Literal["development", "test"]
) -> tuple[Metric, ...]:
    """Reject missing, duplicate or non-finite evaluation cells."""
    protocol = Protocol.model_validate_json((root / "protocol.json").read_bytes())
    rows = tuple(Metric.model_validate(row) for row in evaluate(root, strategy, phase))
    expected = {
        (item.name, budget)
        for item in protocol.files
        if item.role == phase
        for budget in protocol.budgets_seconds
    }
    observed = {(row.session, row.budget_seconds) for row in rows}
    if observed != expected or len(rows) != len(expected):
        message = "Evaluation must return each declared session/budget exactly once"
        raise ResearchError(message)
    if any(row.mode != strategy.mode for row in rows):
        message = "Evaluation returned a different strategy mode"
        raise ResearchError(message)
    return rows


def objective(rows: tuple[Metric, ...]) -> float:
    """Use the predeclared positive-budget development macro mean."""
    return fmean(row.r2 for row in rows if row.budget_seconds in (5, 15, 30))


def verify_history(root: Path, events: tuple[Trial, ...]) -> None:
    """Reject changes to code, protocol, dependencies or runtime during a search."""
    current = fingerprints(root)
    runtime = environment()
    if any(event.inputs != current or event.environment != runtime for event in events):
        message = "Immutable inputs or runtime changed; start a separate research run"
        raise ResearchError(message)


def record_trial(
    root: Path, config_path: Path, hypothesis: str, *, max_trials: int = MAX_TRIALS
) -> Path:
    """Run one development-only hypothesis, preserving starts and all failures."""
    strategy = Strategy.model_validate_json(config_path.read_bytes())
    if strategy.mode == "supervised":
        message = "supervised is a predeclared final reference, not a search candidate"
        raise ResearchError(message)
    if not hypothesis.strip() or not 1 <= max_trials <= MAX_TRIALS:
        message = "A hypothesis and a trial budget between 1 and 12 are required"
        raise ResearchError(message)
    with exclusive_run(root) as results:
        if (results / "frozen.json").exists() or (results / "final.started.json").exists():
            message = "Search stopped: winner selection or final testing has begun"
            raise ResearchError(message)
        ledger = results / "experiments.jsonl"
        events = read_trials(ledger)
        verify_history(root, events)
        if events and any(event.max_trials != max_trials for event in events):
            message = "The recorded trial budget cannot change during a search"
            raise ResearchError(message)
        identifier = digest(strategy)
        previous = [event for event in events if event.trial_id == identifier]
        trial_path = results / "trials" / identifier / "result.json"
        if previous:
            if previous[-1].status in ("keep", "discard"):
                return trial_path
            message = "Candidate already crashed or was interrupted; inspect its ledger"
            raise ResearchError(message)
        if len({event.trial_id for event in events}) >= max_trials:
            message = "The predeclared trial budget is exhausted"
            raise ResearchError(message)
        trial = Trial(
            sequence=len(events) + 1,
            trial_id=identifier,
            strategy=strategy,
            hypothesis=hypothesis.strip(),
            inputs=fingerprints(root),
            environment=environment(),
            max_trials=max_trials,
            started_at=datetime.now(UTC).isoformat(),
            status="running",
        )
        write_record(trial_path.with_name("start.json"), trial, exclusive=True)
        append_trial(ledger, trial)
        try:
            rows = checked_rows(root, strategy, "development")
            score = objective(rows)
            best = max(
                (event.score for event in events if event.score is not None), default=float("-inf")
            )
            completed = trial.model_copy(
                update={
                    "sequence": trial.sequence + 1,
                    "rows": rows,
                    "score": score,
                    "status": "keep" if score > best else "discard",
                }
            )
            verify_history(root, (trial,))
            write_record(trial_path, completed)
            append_trial(ledger, completed)
            return trial_path
        finally:
            error = sys.exception()
            if error is not None:
                crashed = trial.model_copy(
                    update={
                        "sequence": trial.sequence + 1,
                        "status": "crash",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                write_record(trial_path, crashed)
                append_trial(ledger, crashed)


def freeze(root: Path) -> Path:
    """Select the development winner once and predeclare all final comparisons."""
    with exclusive_run(root) as results:
        path = results / "frozen.json"
        if path.exists() or (results / "final.started.json").exists():
            message = "A frozen research lock already exists"
            raise ResearchError(message)
        ledger = results / "experiments.jsonl"
        events = read_trials(ledger)
        verify_history(root, events)
        successful = [event for event in events if event.score is not None]
        if not successful or any(
            event.status == "running"
            for event in {event.trial_id: event for event in events}.values()
        ):
            message = "Freeze requires completed trials and no interrupted trial"
            raise ResearchError(message)
        winner = max(successful, key=lambda event: objective(event.rows))
        write_record(
            path,
            Frozen(
                inputs=fingerprints(root),
                ledger_sha256=file_digest(ledger),
                winner=winner,
                baselines=BASELINES,
                matched_frozen=winner.strategy.model_copy(update={"mode": "frozen"}),
            ),
            exclusive=True,
        )
        return path


def final_test(root: Path) -> Path:
    """Evaluate only the frozen comparison; interruptions resume that same lock."""
    with exclusive_run(root) as results:
        frozen_path = results / "frozen.json"
        frozen = Frozen.model_validate_json(frozen_path.read_bytes())
        if frozen.inputs != fingerprints(root):
            message = "Immutable inputs changed after winner selection"
            raise ResearchError(message)
        if frozen.ledger_sha256 != file_digest(results / "experiments.jsonl"):
            message = "Experiment ledger changed after winner selection"
            raise ResearchError(message)
        verify_history(root, (frozen.winner,))
        frozen_hash = file_digest(frozen_path)
        marker = results / "final.started.json"
        if marker.exists():
            if Started.model_validate_json(marker.read_bytes()).frozen_sha256 != frozen_hash:
                message = "Final testing already started under a different lock"
                raise ResearchError(message)
        else:
            write_record(marker, Started(frozen_sha256=frozen_hash), exclusive=True)
        reports: list[FinalResult] = []
        for label, strategy in comparisons(frozen):
            path = results / "final" / f"{label}.json"
            if path.exists():
                report = FinalResult.model_validate_json(path.read_bytes())
                if report.strategy != strategy or report.frozen_sha256 != frozen_hash:
                    message = "A final result belongs to a different frozen comparison"
                    raise ResearchError(message)
            else:
                report = FinalResult(
                    label=label,
                    strategy=strategy,
                    frozen_sha256=frozen_hash,
                    rows=checked_rows(root, strategy, "test"),
                )
                verify_history(root, (frozen.winner,))
                write_record(path, report)
            reports.append(report)
        report_path = results / "final" / "report.json"
        write_record(report_path, FinalReport(frozen_sha256=frozen_hash, results=tuple(reports)))
        with report_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["label", *Metric.model_fields])
            writer.writeheader()
            for report in reports:
                writer.writerows({"label": report.label, **row.model_dump()} for row in report.rows)
        return report_path
