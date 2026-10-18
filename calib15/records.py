"""Validated audit records and replay for the calibration study."""

import os
from pathlib import Path
from statistics import fmean
from typing import Final, Literal, assert_never

from calib15.io import read_study, results_dir, study_dir
from calib15.schema import Calibration, Features, Row
from neurocalib.ledger import Record, ResearchError, Started, digest, environment, file_digest

BASELINE_ALPHAS: Final = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
BASELINES: Final = tuple(
    Calibration(mode="target", alpha=alpha) for alpha in BASELINE_ALPHAS
) + tuple(
    Calibration(mode="residual", alpha=alpha, strength=strength)
    for alpha in BASELINE_ALPHAS
    for strength in (0.25, 0.5, 1.0)
)
EPSILON: Final = 1e-12
DEVELOPMENT_DAYS: Final = 9
DEFAULT_COMPONENTS: Final = 8
FINAL_METHODS: Final = 4


class Trial(Record):
    """Complete start or terminal snapshot of a development-only experiment."""

    sequence: int
    trial_id: str
    config: Calibration
    features: Features
    hypothesis: str
    kind: Literal["baseline", "candidate"]
    inputs: dict[str, str]
    environment: dict[str, str]
    started_at: str
    status: Literal["running", "keep", "discard", "crash"]
    rows: tuple[Row, ...] = ()
    score: float | None = None
    error: str | None = None


class Gate(Record):
    """Development decision; no external-animal or held-out outcomes enter it."""

    passed: bool
    reason: str
    inputs: dict[str, str]
    ledger_sha256: str
    attempted_candidates: int
    winner: Calibration | None = None
    target_best: Calibration | None = None
    residual_best: Calibration | None = None
    baseline_best: Calibration | None = None
    winner_score: float | None = None
    baseline_score: float | None = None
    delta: float | None = None
    winner_negative_days: int | None = None
    baseline_negative_days: int | None = None


class Comparison(Record):
    """A configuration selected before final-test access."""

    label: str
    config: Calibration


class Frozen(Record):
    """Immutable feature choice, development evidence, and four final methods."""

    decision: Gate
    features: Features
    inputs: dict[str, str]
    ledger_sha256: str
    environment: dict[str, str]
    comparisons: tuple[Comparison, ...]


class Run(Record):
    """One frozen comparison's results, reusable after a failed final run."""

    label: str
    config: Calibration
    frozen_sha256: str
    rows: tuple[Row, ...]


class AnimalOutcome(Record):
    """Primary-budget evidence kept separate for each animal."""

    animal: str
    delta: float
    winner_negative_days: int
    baseline_negative_days: int


class Outcome(Record):
    """Fixed final success rule, with no test-time method reselection."""

    passed: bool
    macro_delta: float
    animals: tuple[AnimalOutcome, ...]


class Report(Record):
    """Completed final or reproduction evidence, including the actual runtime."""

    frozen_sha256: str
    environment: dict[str, str]
    runs: tuple[Run, ...]
    outcome: Outcome
    matches_original: bool | None = None


def fingerprints(root: Path) -> dict[str, str]:
    """Bind code, source-only tuning, scientific instructions, and dependencies."""
    paths = [
        path for package in ("calib15", "neurocalib") for path in (root / package).rglob("*.py")
    ]
    paths.extend(root / name for name in ("pyproject.toml", "uv.lock"))
    paths.extend(study_dir(root) / name for name in ("protocol.json", "PROTOCOL.md", "program.md"))
    paths.extend(results_dir(root) / name for name in ("features.json", "source_tuning.json"))
    return {path.relative_to(root).as_posix(): file_digest(path) for path in sorted(paths)}


def check_history(root: Path, events: tuple[Trial, ...]) -> None:
    """Reject mixing scores from different code, protocols, features, or runtimes."""
    current, runtime = fingerprints(root), environment()
    if any(event.inputs != current or event.environment != runtime for event in events):
        message = "Immutable inputs or runtime changed; do not restart this search"
        raise ResearchError(message)


def read_features(root: Path) -> Features:
    """Load the source-only feature choice established before the first trial."""
    return Features.model_validate_json((results_dir(root) / "features.json").read_bytes())


def read_frozen(root: Path) -> Frozen:
    """Verify the seal and its four declared methods before any target file access."""
    results = results_dir(root)
    if not (results / "frozen.json").exists():
        message = "Final access requires a passing, frozen development decision"
        raise ResearchError(message)
    frozen = Frozen.model_validate_json((results / "frozen.json").read_bytes())
    decision = frozen.decision
    expected = tuple(
        Comparison(label=label, config=config)
        for label, config in (
            ("winner", decision.winner),
            ("target_best", decision.target_best),
            ("residual_best", decision.residual_best),
            ("frozen", Calibration(mode="frozen")),
        )
        if config is not None
    )
    if (
        frozen.inputs != fingerprints(root)
        or not decision.passed
        or len(expected) != FINAL_METHODS
        or frozen.features != read_features(root)
        or frozen.comparisons != expected
        or decision.baseline_best not in (decision.target_best, decision.residual_best)
        or decision.inputs != frozen.inputs
        or decision.ledger_sha256 != frozen.ledger_sha256
        or frozen.ledger_sha256 != file_digest(results / "experiments.jsonl")
    ):
        message = "Frozen inputs, ledger or development gate changed"
        raise ResearchError(message)
    return frozen


def require_final_access(root: Path) -> None:
    """Require a passing seal and its persistent marker before opening test data."""
    _ = read_frozen(root)
    results = results_dir(root)
    if not (results / "final.started.json").exists():
        message = "Final access requires its exclusive start marker"
        raise ResearchError(message)
    marker = Started.model_validate_json((results / "final.started.json").read_bytes())
    if marker.frozen_sha256 != file_digest(results / "frozen.json"):
        message = "Final access marker does not match the frozen comparison"
        raise ResearchError(message)


def mean_r2(rows: tuple[Row, ...]) -> float:
    """Give each recording equal weight at the primary budget."""
    return fmean(row.r2 for row in rows)


def negative_days(rows: tuple[Row, ...]) -> int:
    """Count recording-level harm relative to each day's frozen source decoder."""
    return sum(row.r2 < row.frozen_r2 for row in rows)


def append_event(path: Path, event: Trial) -> None:
    """Persist every state transition before another hypothesis may start."""
    with path.open("a", encoding="utf-8") as stream:
        _ = stream.write(event.model_dump_json() + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_events(path: Path) -> tuple[Trial, ...]:
    """Replay starts and terminal states, rejecting altered snapshots or gaps."""
    if not path.exists():
        return ()
    events = tuple(
        Trial.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()
    )
    previous: dict[str, Trial] = {}
    for index, event in enumerate(events, start=1):
        if event.sequence != index or event.trial_id != digest(event.config):
            message = "Invalid ledger sequence or configuration checksum"
            raise ResearchError(message)
        old = previous.get(event.trial_id)
        match event.status:
            case "running":
                if old is not None:
                    message = "A configuration cannot start twice"
                    raise ResearchError(message)
            case "keep" | "discard" | "crash":
                mutable = {"sequence", "status", "rows", "score", "error"}
                if (
                    old is None
                    or old.status != "running"
                    or old.model_dump(exclude=mutable) != event.model_dump(exclude=mutable)
                ):
                    message = "Terminal event has no matching immutable start"
                    raise ResearchError(message)
            case unreachable:
                assert_never(unreachable)
        previous[event.trial_id] = event
    return events


def validate_rows(
    root: Path,
    rows: tuple[Row, ...],
    phase: Literal["development", "test"],
    budgets: tuple[int, ...],
) -> tuple[Row, ...]:
    """Require every declared recording and budget exactly once."""
    study = read_study(root)
    expected = {
        (item.animal, item.name, budget)
        for item in study.files
        if item.role == phase
        for budget in budgets
    }
    actual = {(row.animal, row.session, row.budget_seconds) for row in rows}
    if expected != actual or len(rows) != len(expected):
        message = "Evaluation cells differ from the pinned recording/budget contract"
        raise ResearchError(message)
    if phase == "development" and (
        len(rows) != DEVELOPMENT_DAYS or any(row.animal != "indy" for row in rows)
    ):
        message = "Development selection requires exactly nine Indy recording days"
        raise ResearchError(message)
    return rows
