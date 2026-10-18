"""Immutable ancestry, finite manifest access, and durable cumulative trial records."""

import os
from pathlib import Path

from calib15.records import validate_rows
from calibnext.ancestry import archive_files, archived_events, read_inheritance, read_log
from calibnext.schema import (
    Batch,
    Config,
    Frozen,
    Inheritance,
    Sanity,
    Trial,
    check_parity,
    decide,
    passes,
)
from neurocalib.ledger import ResearchError, Started, environment, file_digest

__all__ = [
    "append_event",
    "archive_files",
    "batch_path",
    "check_history",
    "check_sanity",
    "fingerprints",
    "method_paths",
    "read_batch",
    "read_events",
    "read_frozen",
    "read_inheritance",
    "require_final_access",
    "results_dir",
    "study_dir",
]


def study_dir(root: Path) -> Path:
    """Keep additional batches separate from every already bound study."""
    return root / "studies" / "calibnext"


def results_dir(root: Path) -> Path:
    """Return the cumulative follow-up evidence directory."""
    return study_dir(root) / "results"


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


def check_sanity(root: Path, inherited: Inheritance) -> None:
    """Require verified pipeline parity before another development label is exposed."""
    sanity = Sanity.model_validate_json((results_dir(root) / "sanity.json").read_bytes())
    _ = validate_rows(root, sanity.rows, "development", (15,))
    check_parity(sanity, inherited)


def method_paths(root: Path, configs: tuple[Config, ...]) -> tuple[Path, ...]:
    """Resolve custom leaf scripts without accepting symlinks outside the method folder."""
    root = root.resolve()
    paths = tuple(
        (root / "methods" / f"{config.method}.py").resolve() for config in configs if config.method
    )
    if any(path.parent != root / "methods" for path in paths):
        message = "A method must be a leaf script inside the repository methods folder"
        raise ResearchError(message)
    return paths


def fingerprints(root: Path, configs: tuple[Config, ...] = ()) -> dict[str, str]:
    """Bind fixed core and only selected scripts; unused methods remain appendable."""
    root = root.resolve()
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
    paths.extend(method_paths(root, configs))
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
    return archived_events(root) if archive else read_log(results_dir(root) / "experiments.jsonl")


def check_history(root: Path, events: tuple[Trial, ...]) -> Inheritance:
    """Detect source, manifest or environment edits instead of hiding prior exposure."""
    inherited = read_inheritance(root)
    check_sanity(root, inherited)
    runtime = environment()
    if any(
        event.inputs != fingerprints(root, (event.config,))
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
        frozen.inputs != fingerprints(root, tuple(event.config for event in events))
        or any(event.inputs != fingerprints(root, (event.config,)) for event in events)
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
