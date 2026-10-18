"""Durable experiment records and checksums for the fixed research contract."""

import hashlib
import json
import os
import platform
import sys
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import ClassVar, Final, Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field

from neurocalib.schema import Mode, Strategy

OBJECTIVE: Final = "mean development R2 over sessions and budgets 5, 15, 30 seconds"
MAX_TRIALS: Final = 12
BASELINES: Final = tuple(
    Strategy(mode=mode) for mode in ("frozen", "center", "standardize", "coral", "supervised")
)


class ResearchError(RuntimeError):
    """An experiment would violate its recorded research contract."""


class Record(BaseModel):
    """Validated, finite JSON records with no undeclared fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False
    )


class Metric(Record):
    """One session and calibration budget from the fixed evaluator."""

    session: str
    mode: Mode
    budget_seconds: int = Field(ge=0)
    r2: float
    rmse: float = Field(ge=0)
    inference_us: float = Field(ge=0)
    calibration_seconds: float = Field(ge=0)
    missing_channels: int = Field(ge=0)
    r2_x: float
    r2_y: float
    zero_r2: float
    zero_rmse: float = Field(ge=0)
    n_scoring_bins: int = Field(gt=0)
    n_prefix_bins: int = Field(ge=0)
    n_channels: int = Field(gt=0)
    n_features: int = Field(gt=0)
    last_bin_seconds: float = Field(gt=0)


class Trial(Record):
    """A full trial snapshot, appended once at start and once at completion."""

    sequence: int
    trial_id: str
    strategy: Strategy
    hypothesis: str = Field(min_length=1)
    inputs: dict[str, str]
    environment: dict[str, str]
    max_trials: int = Field(ge=1, le=12)
    started_at: str
    status: Literal["running", "keep", "discard", "crash"]
    score: float | None = None
    error: str | None = None
    rows: tuple[Metric, ...] = ()


class Frozen(Record):
    """Exclusive winner selection made before any final evaluation."""

    inputs: dict[str, str]
    ledger_sha256: str
    winner: Trial
    baselines: tuple[Strategy, ...]
    matched_frozen: Strategy
    objective: str = OBJECTIVE


class FinalResult(Record):
    """A frozen strategy's final results, reusable after an interruption."""

    label: str
    strategy: Strategy
    frozen_sha256: str
    rows: tuple[Metric, ...]


class FinalReport(Record):
    """The complete predeclared final comparison."""

    frozen_sha256: str
    results: tuple[FinalResult, ...]


class Started(Record):
    """Persistent evidence that final-test access has begun."""

    frozen_sha256: str


def comparisons(frozen: Frozen) -> tuple[tuple[str, Strategy], ...]:
    """Expose only the winner and comparisons declared before final-test access."""
    return (
        ("winner", frozen.winner.strategy),
        *((f"baseline_{item.mode}", item) for item in frozen.baselines),
        ("matched_frozen", frozen.matched_frozen),
    )


def digest(model: BaseModel) -> str:
    """Hash validated content independently of JSON whitespace and key order."""
    payload = json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_digest(path: Path) -> str:
    """Hash a file without loading recordings into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprints(root: Path) -> dict[str, str]:
    """Pin code, protocol and dependencies without opening held-out recordings."""
    paths = sorted((root / "neurocalib").rglob("*.py"))
    paths.extend(
        root / name for name in ("protocol.json", "PROTOCOL.md", "pyproject.toml", "uv.lock")
    )
    return {path.relative_to(root).as_posix(): file_digest(path) for path in paths}


def environment() -> dict[str, str]:
    """Capture the actual numerical runtime and thread policy."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "h5py": version("h5py"),
        "pydantic": version("pydantic"),
        "randomness": "deterministic closed-form methods; no stochastic trial seed",
        **{
            name: os.environ.get(name, "unset")
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def write_record(path: Path, record: BaseModel, *, exclusive: bool = False) -> None:
    """Write snapshots atomically; research locks are exclusively created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            _ = stream.write(record.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        _ = temporary.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        _ = temporary.replace(path)


def append_trial(path: Path, trial: Trial) -> None:
    """Append and flush each state transition before continuing."""
    with path.open("a", encoding="utf-8") as stream:
        _ = stream.write(trial.model_dump_json() + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_trials(path: Path) -> tuple[Trial, ...]:
    """Replay the complete ledger, rejecting gaps and invalid transitions."""
    if not path.exists():
        return ()
    events = tuple(
        Trial.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()
    )
    latest: dict[str, Trial] = {}
    for sequence, event in enumerate(events, start=1):
        if event.sequence != sequence or digest(event.strategy) != event.trial_id:
            message = "Ledger sequence or candidate checksum is invalid"
            raise ResearchError(message)
        previous = latest.get(event.trial_id)
        match event.status:
            case "running":
                if previous is not None:
                    message = "Ledger repeats a candidate start"
                    raise ResearchError(message)
            case "keep" | "discard" | "crash":
                if previous is None or previous.status != "running":
                    message = "Ledger terminal state has no unmatched start"
                    raise ResearchError(message)
                mutable = {"sequence", "status", "score", "error", "rows"}
                if event.model_dump(exclude=mutable) != previous.model_dump(exclude=mutable):
                    message = "Ledger altered an experiment snapshot"
                    raise ResearchError(message)
            case unreachable:
                assert_never(unreachable)
        latest[event.trial_id] = event
    return events


@contextmanager
def exclusive_run(root: Path) -> Generator[Path]:
    """Serialize local research commands; stale locks require explicit inspection."""
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    active = results / ".active"
    # ponytail: one local process; use a database transaction if distributed runs are needed.
    with active.open("x", encoding="utf-8") as stream:
        _ = stream.write(f"pid={os.getpid()} started={datetime.now(UTC).isoformat()}\n")
    try:
        yield results
    finally:
        active.unlink()
