"""Bounded development hypotheses for the separately frozen calibration study."""

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from calib15.evaluation import evaluate
from calib15.io import read_study, results_dir, study_dir
from calib15.records import (
    BASELINES,
    DEFAULT_COMPONENTS,
    EPSILON,
    Comparison,
    Frozen,
    Gate,
    Trial,
    append_event,
    check_history,
    fingerprints,
    mean_r2,
    negative_days,
    read_events,
    read_features,
    require_final_access,
    validate_rows,
)
from calib15.replay import reproduce_development
from calib15.schema import Calibration
from calib15.seal import final
from calib15.seal import reproduce as reproduce_final
from neurocalib.ledger import (
    ResearchError,
    digest,
    environment,
    exclusive_run,
    file_digest,
    write_record,
)

__all__ = (
    "ResearchError",
    "final",
    "freeze",
    "gate",
    "reproduce",
    "require_final_access",
    "run_trial",
)


def run_trial(
    root: Path, config: Calibration, hypothesis: str, kind: Literal["baseline", "candidate"]
) -> Path:
    """Evaluate one hypothesis on only the nine declared development days."""
    study = read_study(root)
    if (
        not hypothesis.strip()
        or ((kind == "baseline") != (config in BASELINES))
        or (kind == "candidate" and config.mode not in ("affine", "subspace"))
        or (kind == "candidate" and config.strength <= 0)
        or (config.mode == "affine" and config.components != DEFAULT_COMPONENTS)
    ):
        message = "Use declared baselines or affine/subspace hypotheses with a stated rationale"
        raise ResearchError(message)
    with exclusive_run(study_dir(root)) as results:
        if (results / "frozen.json").exists() or (results / "final.started.json").exists():
            message = "Search is permanently stopped after freezing or final access"
            raise ResearchError(message)
        ledger = results / "experiments.jsonl"
        events = read_events(ledger)
        check_history(root, events)
        identifier = digest(config)
        prior = [event for event in events if event.trial_id == identifier]
        path = results / "trials" / identifier / "result.json"
        if prior:
            if prior[-1].status in ("keep", "discard"):
                return path
            message = "Crashed or interrupted configurations cannot restart"
            raise ResearchError(message)
        complete = [event for event in events if event.score is not None]
        if kind == "candidate" and not all(
            any(event.config == baseline for event in complete) for baseline in BASELINES
        ):
            message = "Complete every strong baseline before proposing candidates"
            raise ResearchError(message)
        attempted = {event.trial_id for event in events if event.kind == "candidate"}
        if kind == "candidate" and len(attempted) >= study.max_candidates:
            message = "The fixed candidate budget is exhausted"
            raise ResearchError(message)
        trial = Trial(
            sequence=len(events) + 1,
            trial_id=identifier,
            config=config,
            features=read_features(root),
            hypothesis=hypothesis.strip(),
            kind=kind,
            inputs=fingerprints(root),
            environment=environment(),
            started_at=datetime.now(UTC).isoformat(),
            status="running",
        )
        write_record(path.with_name("start.json"), trial, exclusive=True)
        append_event(ledger, trial)
        try:
            rows = validate_rows(
                root,
                evaluate(root, trial.features, config, "development"),
                "development",
                (study.primary_budget_seconds,),
            )
            score = mean_r2(rows)
            best = max(
                (
                    event.score
                    for event in complete
                    if event.kind == kind and event.score is not None
                ),
                default=float("-inf"),
            )
            completed = trial.model_copy(
                update={
                    "sequence": trial.sequence + 1,
                    "rows": rows,
                    "score": score,
                    "status": "keep" if score > best else "discard",
                }
            )
            check_history(root, (trial,))
            write_record(path, completed)
            append_event(ledger, completed)
            return path
        finally:
            error = sys.exception()
            if error is not None:
                failed = trial.model_copy(
                    update={
                        "sequence": trial.sequence + 1,
                        "status": "crash",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                write_record(path, failed)
                append_event(ledger, failed)


def gate(root: Path) -> Gate:
    """Record the fixed development decision without accessing final targets."""
    with exclusive_run(study_dir(root)):
        return decide(root)


def reproduce(root: Path) -> Path:
    """Recompute final evidence when complete, otherwise development trials only."""
    if (results_dir(root) / "final" / "report.json").exists():
        return reproduce_final(root)
    return reproduce_development(root)


def decide(root: Path) -> Gate:
    """Compute one decision while its caller holds the local research lock."""
    study, results = read_study(root), results_dir(root)
    ledger = results / "experiments.jsonl"
    events = read_events(ledger)
    check_history(root, events)
    complete = [event for event in events if event.score is not None]
    latest = {event.trial_id: event for event in events}
    attempted = len({event.trial_id for event in events if event.kind == "candidate"})
    decision = Gate(
        passed=False,
        reason="Complete all eight baselines and at least one candidate",
        inputs=fingerprints(root),
        ledger_sha256=file_digest(ledger) if events else "",
        attempted_candidates=attempted,
    )
    if all(
        any(event.config == baseline for event in complete) for baseline in BASELINES
    ) and not any(event.status == "running" for event in latest.values()):
        target = max(
            (event for event in complete if event.config.mode == "target"),
            key=lambda event: mean_r2(event.rows),
        )
        residual = max(
            (event for event in complete if event.config.mode == "residual"),
            key=lambda event: mean_r2(event.rows),
        )
        baseline = max((target, residual), key=lambda event: mean_r2(event.rows))
        candidates = [event for event in complete if event.kind == "candidate"]
        eligible = [
            event
            for event in candidates
            if mean_r2(event.rows) - mean_r2(baseline.rows) + EPSILON >= study.gate_delta
            and negative_days(event.rows) <= negative_days(baseline.rows)
        ]
        if candidates:
            winner = max(eligible or candidates, key=lambda event: mean_r2(event.rows))
            decision = decision.model_copy(
                update={
                    "passed": bool(eligible),
                    "reason": "Development gate passed"
                    if eligible
                    else "Candidate gain or negative-transfer count failed the development gate",
                    "winner": winner.config,
                    "target_best": target.config,
                    "residual_best": residual.config,
                    "baseline_best": baseline.config,
                    "winner_score": mean_r2(winner.rows),
                    "baseline_score": mean_r2(baseline.rows),
                    "delta": mean_r2(winner.rows) - mean_r2(baseline.rows),
                    "winner_negative_days": negative_days(winner.rows),
                    "baseline_negative_days": negative_days(baseline.rows),
                }
            )
    write_record(results / "decision.json", decision)
    return decision


def freeze(root: Path) -> Path:
    """Seal a passing decision before either final animal contributes target data."""
    with exclusive_run(study_dir(root)) as results:
        path = results / "frozen.json"
        if path.exists() or (results / "final.started.json").exists():
            message = "An exclusive final research seal already exists"
            raise ResearchError(message)
        decision = decide(root)
        if (
            not decision.passed
            or decision.winner is None
            or decision.target_best is None
            or decision.residual_best is None
        ):
            message = "Failed development gate: final testing is not authorized"
            raise ResearchError(message)
        comparisons = (
            Comparison(label="winner", config=decision.winner),
            Comparison(label="target_best", config=decision.target_best),
            Comparison(label="residual_best", config=decision.residual_best),
            Comparison(label="frozen", config=Calibration(mode="frozen")),
        )
        write_record(
            path,
            Frozen(
                decision=decision,
                features=read_features(root),
                inputs=fingerprints(root),
                ledger_sha256=decision.ledger_sha256,
                environment=environment(),
                comparisons=comparisons,
            ),
            exclusive=True,
        )
        return path
