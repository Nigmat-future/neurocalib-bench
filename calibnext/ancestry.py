"""Read-only historical evidence and the explicit ordered continuation chain."""

import math
from pathlib import Path
from typing import ClassVar, Final, assert_never

from pydantic import ConfigDict, Field

from calib15.records import Gate as ParentGate
from calib15.records import mean_r2, negative_days, validate_rows
from calib15.records import read_events as read_parent_events
from calib15.schema import Calibration, Features
from calibnext.schema import EPSILON, Decision, Inheritance, Sanity, Trial, previous_attempts
from neurocalib.ledger import Record, ResearchError, digest, file_digest

ANCHOR_SCORE: Final = 0.35010946184569786
ANCHOR_HARM: Final = 1
INHERITED_CONFIGURATIONS: Final = 32
INHERITED_CANDIDATES: Final = 8
ARCHIVES: Final[tuple[tuple[str, int, tuple[str, ...]], ...]] = (
    ("calibnext-map-initial", 8, ("map",)),
    ("calibnext-linear-encoding", 30, ("pls", "encoding")),
)


class Provenance(Record):
    """Read declared exposure while allowing the protocol's other fixed metadata."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="ignore", allow_inf_nan=False
    )
    previous_continuation_attempts: int | None = Field(default=None, ge=0, strict=True)


def archive_folders(root: Path) -> tuple[Path, ...]:
    """Permit only a complete prefix of the fixed historical chain."""
    expected = tuple(root / "studies" / name for name, _, _ in ARCHIVES)
    present = tuple(path for path in expected if path.exists())
    if present != expected[: len(present)]:
        message = "Continuation archives must retain their complete preceding chain"
        raise ResearchError(message)
    return present


def archive_files(root: Path) -> tuple[Path, ...]:
    """Return the four immutable evidence files for every declared historical stage."""
    paths = tuple(
        path
        for folder in archive_folders(root)
        for path in (
            folder / "protocol.json",
            *(
                folder / "results" / name
                for name in ("experiments.jsonl", "decision.json", "sanity.json")
            ),
        )
    )
    if not all(path.is_file() for path in paths):
        message = "Continuation archive must retain all four complete evidence files"
        raise ResearchError(message)
    return paths


def read_log(path: Path) -> tuple[Trial, ...]:
    """Replay complete immutable start/end snapshots without executing algorithms."""
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


def archived_events(root: Path) -> tuple[Trial, ...]:
    """Read independent archived ledgers for cross-stage duplicate detection."""
    _ = archive_files(root)
    return tuple(
        event
        for folder in archive_folders(root)
        for event in read_log(folder / "results" / "experiments.jsonl")
    )


def read_inheritance(root: Path) -> Inheritance:
    """Retain the fixed baseline and accumulate the ordered failed continuations."""
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
    _ = archive_files(root)
    seen: set[str] = set()
    for folder, (_, count, modes) in zip(archive_folders(root), ARCHIVES, strict=False):
        previous = read_log(folder / "results" / "experiments.jsonl")
        identifiers = {event.trial_id for event in previous}
        if seen & identifiers:
            message = "A configuration was repeated across continuation archives"
            raise ResearchError(message)
        if any(event.config.mode not in modes for event in previous):
            message = "Unexpected method in a historical continuation archive"
            raise ResearchError(message)
        added = previous_attempts(
            previous,
            Decision.model_validate_json((folder / "results" / "decision.json").read_bytes()),
            Sanity.model_validate_json((folder / "results" / "sanity.json").read_bytes()),
            inherited,
            expected_count=count,
        )
        seen.update(identifiers)
        inherited = inherited.model_copy(
            update={
                "previous_continuation_attempts": inherited.previous_continuation_attempts + added
            }
        )
    metadata = Provenance.model_validate_json(
        (root / "studies" / "calibnext" / "protocol.json").read_bytes()
    )
    if (
        metadata.previous_continuation_attempts is not None
        and metadata.previous_continuation_attempts != inherited.previous_continuation_attempts
    ):
        message = "Declared continuation exposure differs from the complete archive chain"
        raise ResearchError(message)
    return inherited
