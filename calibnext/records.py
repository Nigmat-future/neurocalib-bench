"""Immutable ancestry, finite manifest access, and durable cumulative trial records."""

import math
import os
from pathlib import Path
from typing import Final, assert_never

from calib15.records import Gate as ParentGate
from calib15.records import mean_r2, negative_days, validate_rows
from calib15.records import read_events as read_parent_events
from calib15.schema import Calibration, Features
from calibnext.schema import (
    EPSILON,
    Batch,
    Decision,
    Frozen,
    Inheritance,
    Sanity,
    Trial,
    check_parity,
    decide,
    passes,
    previous_attempts,
)
from neurocalib.ledger import ResearchError, Started, digest, environment, file_digest

ANCHOR_SCORE: Final = 0.35010946184569786
ANCHOR_HARM: Final = 1
INHERITED_CONFIGURATIONS: Final = 32
INHERITED_CANDIDATES: Final = 8


def study_dir(root: Path) -> Path:
    """Keep additional batches separate from every already bound study."""
    return root / "studies" / "calibnext"


def results_dir(root: Path) -> Path:
    """Return the cumulative follow-up evidence directory."""
    return study_dir(root) / "results"


def archive_files(root: Path) -> tuple[Path, ...]:
    """Bind the explicit historical continuation when its archive is present."""
    archive = root / "studies" / "calibnext-map-initial"
    return (
        (
            archive / "protocol.json",
            *(
                archive / "results" / name
                for name in ("experiments.jsonl", "decision.json", "sanity.json")
            ),
        )
        if archive.exists()
        else ()
    )


def batch_path(root: Path, batch_id: str) -> Path:
    """Resolve only validated manifest identities inside the study directory."""
    if not batch_id or any(
        character not in "0123456789abcdefghijklmnopqrstuvwxyz-" for character in batch_id
    ):
        message = "Unsafe batch identity"
        raise ResearchError(message)
    return study_dir(root) / "batches" / f"{batch_id}.json"


def read_batch(root: Path, batch_id: str) -> Batch:
    """Read the finite declaration without executing any experiment."""
    batch = Batch.model_validate_json(batch_path(root, batch_id).read_bytes())
    if batch.batch_id != batch_id:
        message = "Batch filename and declared identity differ"
        raise ResearchError(message)
    return batch


def read_inheritance(root: Path) -> Inheritance:
    """Recompute ancestry and the fixed reference from already exposed parent rows."""
    parent = root / "studies" / "calib15" / "results"
    ledger = parent / "experiments.jsonl"
    events = read_parent_events(ledger)
    decision = ParentGate.model_validate_json((parent / "decision.json").read_bytes())
    baseline = next(
        (
            event
            for event in events
            if event.score is not None and event.config == decision.baseline_best
        ),
        None,
    )
    features = Features.model_validate_json((parent / "features.json").read_bytes())
    if (
        baseline is None
        or decision.target_best is None
        or decision.baseline_best is None
        or decision.passed
        or decision.ledger_sha256 != file_digest(ledger)
        or len({event.trial_id for event in events}) != INHERITED_CONFIGURATIONS
        or len({event.trial_id for event in events if event.kind == "candidate"})
        != INHERITED_CANDIDATES
    ):
        message = "Parent evidence must retain the failed 32-configuration, 8-candidate study"
        raise ResearchError(message)
    _ = validate_rows(root, baseline.rows, "development", (15,))
    if (
        not math.isclose(mean_r2(baseline.rows), ANCHOR_SCORE, abs_tol=EPSILON, rel_tol=0)
        or baseline.features != features
        or negative_days(baseline.rows) != ANCHOR_HARM
        or decision.baseline_negative_days != ANCHOR_HARM
        or decision.baseline_score != ANCHOR_SCORE
        or any(file_digest(root / name) != checksum for name, checksum in baseline.inputs.items())
    ):
        message = "Inherited baseline score, harm, features or scientific inputs changed"
        raise ResearchError(message)
    inherited = Inheritance(
        baseline=baseline,
        features=features,
        parent_ledger_sha256=file_digest(ledger),
        references=(decision.baseline_best, decision.target_best, Calibration(mode="frozen")),
    )
    archived = archive_files(root)
    if archived:
        if not all(path.is_file() for path in archived):
            message = "Continuation archive must retain all four complete evidence files"
            raise ResearchError(message)
        count = previous_attempts(
            read_events(root, archive=True),
            Decision.model_validate_json(archived[2].read_bytes()),
            Sanity.model_validate_json(archived[3].read_bytes()),
            inherited,
        )
        inherited = inherited.model_copy(update={"previous_continuation_attempts": count})
    return inherited


def check_sanity(root: Path, inherited: Inheritance) -> None:
    """Require verified pipeline parity before another development label is exposed."""
    sanity = Sanity.model_validate_json((results_dir(root) / "sanity.json").read_bytes())
    _ = validate_rows(root, sanity.rows, "development", (15,))
    check_parity(sanity, inherited)


def fingerprints(root: Path) -> dict[str, str]:
    """Bind code and ancestry without treating future manifests as old trial inputs."""
    paths = [
        path
        for package in ("calibnext", "calib15", "neurocalib")
        for path in (root / package).rglob("*.py")
    ]
    paths.extend(root / name for name in ("pyproject.toml", "uv.lock"))
    for folder in (study_dir(root), root / "studies" / "calib15"):
        paths.extend(folder / name for name in ("protocol.json", "PROTOCOL.md", "program.md"))
    parent = root / "studies" / "calib15" / "results"
    paths.extend(
        parent / name
        for name in ("features.json", "source_tuning.json", "experiments.jsonl", "decision.json")
    )
    paths.append(results_dir(root) / "sanity.json")
    paths.extend(archive_files(root))
    paths.extend(
        path
        for path in (study_dir(root) / "diagnose.py", results_dir(root) / "diagnosis.json")
        if path.exists()
    )
    return {path.relative_to(root).as_posix(): file_digest(path) for path in sorted(paths)}


def append_event(path: Path, event: Trial) -> None:
    """Flush every start and terminal state before advancing the declared batch."""
    with path.open("a", encoding="utf-8") as stream:
        _ = stream.write(event.model_dump_json() + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_events(root: Path, *, archive: bool = False) -> tuple[Trial, ...]:
    """Replay immutable snapshots, including unsuccessful and crashed hypotheses."""
    folder = (
        root / "studies" / "calibnext-map-initial" / "results" if archive else results_dir(root)
    )
    path = folder / "experiments.jsonl"
    if not path.exists():
        return ()
    events = tuple(
        Trial.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()
    )
    latest: dict[str, Trial] = {}
    for sequence, event in enumerate(events, start=1):
        if event.sequence != sequence or event.trial_id != digest(event.config):
            message = "Invalid cumulative ledger sequence or candidate checksum"
            raise ResearchError(message)
        previous = latest.get(event.trial_id)
        match event.status:
            case "running":
                if previous is not None:
                    message = "A cumulative configuration cannot start twice"
                    raise ResearchError(message)
            case "keep" | "discard" | "crash":
                mutable = {"sequence", "status", "score", "rows", "error"}
                if (
                    previous is None
                    or previous.status != "running"
                    or previous.model_dump(exclude=mutable) != event.model_dump(exclude=mutable)
                ):
                    message = "Terminal event has no matching immutable start"
                    raise ResearchError(message)
            case unreachable:
                assert_never(unreachable)
        latest[event.trial_id] = event
    return events


def check_history(root: Path, events: tuple[Trial, ...]) -> Inheritance:
    """Detect source, manifest or environment edits instead of hiding prior exposure."""
    inherited = read_inheritance(root)
    check_sanity(root, inherited)
    current, runtime = fingerprints(root), environment()
    if any(
        event.inputs != current
        or event.environment != runtime
        or event.batch_sha256 != file_digest(batch_path(root, event.batch_id))
        for event in events
    ):
        message = "Bound scientific inputs, batch manifest or runtime changed"
        raise ResearchError(message)
    return inherited


def read_frozen(root: Path) -> Frozen:
    """Verify winner, ancestry, and every used manifest before final data access."""
    path = results_dir(root) / "frozen.json"
    if not path.exists():
        message = "Final access requires a passing cumulative research freeze"
        raise ResearchError(message)
    frozen = Frozen.model_validate_json(path.read_bytes())
    inherited = read_inheritance(root)
    check_sanity(root, inherited)
    events = read_events(root)
    expected_winner = next(
        (event for event in events if event.score is not None and passes(event, inherited)), None
    )
    latest = {event.trial_id: event for event in events}
    if (
        frozen.inputs != fingerprints(root)
        or not frozen.decision.passed
        or frozen.features != inherited.features
        or frozen.baseline != inherited.baseline
        or frozen.references != inherited.references
        or frozen.winner.config != frozen.decision.winner
        or frozen.winner != expected_winner
        or frozen.decision != decide(events, inherited)
        or any(event.status == "running" for event in latest.values())
        or frozen.batch_hashes != {event.batch_id: event.batch_sha256 for event in events}
        or frozen.parent_ledger_sha256 != inherited.parent_ledger_sha256
        or frozen.ledger_sha256 != file_digest(results_dir(root) / "experiments.jsonl")
        or not passes(frozen.winner, inherited)
        or any(
            file_digest(batch_path(root, name)) != checksum
            for name, checksum in frozen.batch_hashes.items()
        )
    ):
        message = "Frozen winner, inherited reference or scientific inputs changed"
        raise ResearchError(message)
    return frozen


def require_final_access(root: Path) -> None:
    """Permit holdout reads only after a passing freeze and its matching start marker."""
    _ = read_frozen(root)
    marker = results_dir(root) / "final.started.json"
    if not marker.exists() or Started.model_validate_json(
        marker.read_bytes()
    ).frozen_sha256 != file_digest(results_dir(root) / "frozen.json"):
        message = "Final access requires the marker for this exact frozen comparison"
        raise ResearchError(message)
