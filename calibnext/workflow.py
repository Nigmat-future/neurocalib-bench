"""Finite follow-up batches on explicitly reused development recordings."""

import sys
from datetime import UTC, datetime
from pathlib import Path

from calib15.records import mean_r2, validate_rows
from calibnext.records import (
    append_event,
    batch_path,
    check_history,
    fingerprints,
    read_batch,
    read_events,
    results_dir,
    study_dir,
)
from calibnext.schema import Batch, Config, Decision, Frozen, Trial, decide, passes
from neurocalib.ledger import (
    ResearchError,
    digest,
    environment,
    exclusive_run,
    file_digest,
    write_record,
)


def check_batch_order(root: Path, batch: Batch, config: Config, events: tuple[Trial, ...]) -> None:
    """Enforce finite order and a completed failed predecessor before another batch."""
    current = [
        event for event in events if event.batch_id == batch.batch_id and event.status == "running"
    ]
    if not current:
        expected_parent = file_digest(results_dir(root) / "experiments.jsonl") if events else ""
        if batch.parent_ledger_sha256 != expected_parent:
            message = "A new batch must extend the exact preceding ledger state"
            raise ResearchError(message)
        if events:
            previous = read_batch(root, events[-1].batch_id)
            completed = {
                event.trial_id
                for event in events
                if event.batch_id == previous.batch_id and event.status != "running"
            }
            if completed != {digest(item) for item in previous.configs}:
                message = "Finish the previous failed finite batch before declaring another"
                raise ResearchError(message)
    elif events[-1].batch_id != batch.batch_id:
        message = "Cannot return to an earlier exposure batch"
        raise ResearchError(message)
    completed_ids = {
        event.trial_id
        for event in events
        if event.batch_id == batch.batch_id and event.status != "running"
    }
    if (
        len(current) != len(completed_ids)
        or len(current) >= len(batch.configs)
        or config != batch.configs[len(current)]
    ):
        message = "Run only the next finite declared configuration; interrupted trials need audit"
        raise ResearchError(message)


def run_trial(root: Path, config: Config, hypothesis: str, batch_id: str) -> Path:
    """Run the next declared hypothesis, stopping immediately after a passing result."""
    from calibnext.evaluation import evaluate  # noqa: PLC0415

    with exclusive_run(study_dir(root)) as results:
        if (results / "frozen.json").exists() or (results / "final.started.json").exists():
            message = "Cumulative research has already been frozen"
            raise ResearchError(message)
        events = read_events(root)
        inherited = check_history(root, events)
        batch = read_batch(root, batch_id)
        identifier = digest(config)
        if any(event.trial_id == identifier for event in read_events(root, archive=True)):
            message = "An archived configuration cannot restart; reproduce its historical commit"
            raise ResearchError(message)
        path = results / "trials" / identifier / "result.json"
        prior = [event for event in events if event.trial_id == identifier]
        if prior:
            if prior[-1].batch_id == batch_id and prior[-1].score is not None:
                write_record(path, prior[-1])
                return path
            message = (
                "Previously attempted or crashed configurations cannot restart in another batch"
            )
            raise ResearchError(message)
        if any(passes(event, inherited) for event in events if event.score is not None):
            message = "A candidate already passed; stop all later batches and freeze"
            raise ResearchError(message)
        if not hypothesis.strip():
            message = "An additional experiment requires a stated hypothesis"
            raise ResearchError(message)
        check_batch_order(root, batch, config, events)
        trial = Trial(
            sequence=len(events) + 1,
            trial_id=identifier,
            batch_id=batch_id,
            batch_sha256=file_digest(batch_path(root, batch_id)),
            config=config,
            features=inherited.features,
            hypothesis=hypothesis.strip(),
            inputs=fingerprints(root, (config,)),
            environment=environment(),
            started_at=datetime.now(UTC).isoformat(),
            status="running",
        )
        write_record(path.with_name("start.json"), trial, exclusive=True)
        ledger = results / "experiments.jsonl"
        append_event(ledger, trial)
        try:
            rows = validate_rows(
                root, evaluate(root, config, "development", budgets=(15,)), "development", (15,)
            )
            score = mean_r2(rows)
            best = max(
                (event.score for event in events if event.score is not None), default=-float("inf")
            )
            done = trial.model_copy(
                update={
                    "sequence": trial.sequence + 1,
                    "rows": rows,
                    "score": score,
                    "status": "keep" if score > best else "discard",
                }
            )
            _ = check_history(root, (trial,))
            write_record(path, done)
            append_event(ledger, done)
            return path
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
                write_record(path, crashed)
                append_event(ledger, crashed)


def gate(root: Path) -> Decision:
    """Write a transparent repeated-development decision without reading holdouts."""
    with exclusive_run(study_dir(root)) as results:
        events = read_events(root)
        inherited = check_history(root, events)
        if any(
            event.status == "running"
            for event in {event.trial_id: event for event in events}.values()
        ):
            message = "An interrupted trial remains in the cumulative ledger"
            raise ResearchError(message)
        decision = decide(events, inherited)
        write_record(results / "decision.json", decision)
        return decision


def freeze(root: Path) -> Path:
    """Create one exclusive final selection retaining the exact earlier strong reference."""
    with exclusive_run(study_dir(root)) as results:
        path = results / "frozen.json"
        if path.exists() or (results / "final.started.json").exists():
            message = "An exclusive cumulative freeze already exists"
            raise ResearchError(message)
        events = read_events(root)
        inherited = check_history(root, events)
        if any(
            event.status == "running"
            for event in {event.trial_id: event for event in events}.values()
        ):
            message = "An interrupted trial remains in the cumulative ledger"
            raise ResearchError(message)
        decision = decide(events, inherited)
        write_record(results / "decision.json", decision)
        if not decision.passed:
            message = "Failed cumulative development gate; holdout access remains blocked"
            raise ResearchError(message)
        winner = next(
            event for event in events if event.score is not None and event.config == decision.winner
        )
        write_record(
            path,
            Frozen(
                winner=winner,
                baseline=inherited.baseline,
                features=inherited.features,
                references=inherited.references,
                decision=decision,
                inputs=fingerprints(root, tuple(event.config for event in events)),
                batch_hashes={event.batch_id: event.batch_sha256 for event in events},
                ledger_sha256=file_digest(results / "experiments.jsonl"),
                parent_ledger_sha256=inherited.parent_ledger_sha256,
                environment=environment(),
            ),
            exclusive=True,
        )
        return path
