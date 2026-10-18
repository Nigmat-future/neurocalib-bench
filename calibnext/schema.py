"""Validated configurations and evidence for explicitly cumulative follow-up batches."""

from typing import Final, Literal, Self, assert_never

from pydantic import Field, model_validator

from calib15.records import Trial as ParentTrial
from calib15.records import mean_r2, negative_days
from calib15.schema import Calibration, Features, Row
from neurocalib.ledger import Record, digest

MAP_COMPONENTS: Final = 8
GATE_MARGIN: Final = 0.02
EPSILON: Final = 1e-12


class Config(Record):
    """The only new search surface; inactive parameters have one canonical value."""

    mode: Literal["map", "pls"] = "map"
    alpha: float = Field(default=10.0, gt=0)
    strength: float = Field(default=1.0, gt=0, le=1)
    components: int = Field(default=8, ge=1, le=32)
    geometry: Literal["identity", "source"] = "identity"
    bias_penalty: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def active_parameters(self) -> Self:
        """Reject fake novelty produced by changing parameters a method ignores."""
        match self.mode:
            case "map":
                valid = self.components == MAP_COMPONENTS
            case "pls":
                valid = self.geometry == "identity" and self.bias_penalty == 0
            case unreachable:
                assert_never(unreachable)
        if not valid:
            message = "map requires components=8; pls requires identity geometry and bias_penalty=0"
            raise ValueError(message)
        return self


class Batch(Record):
    """A finite, ordered declaration made before this batch sees development scores."""

    batch_id: str = Field(pattern=r"^\d{3}-[a-z0-9-]+$")
    parent_ledger_sha256: str = Field(default="", pattern=r"^(?:[a-f0-9]{64})?$")
    rationale: str = Field(min_length=1)
    configs: tuple[Config, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_configs(self) -> Self:
        """Canonical content, rather than formatting or prose, defines an experiment."""
        if len({digest(config) for config in self.configs}) != len(self.configs):
            message = "a batch cannot contain duplicate configurations"
            raise ValueError(message)
        return self


class Inheritance(Record):
    """The already exposed development reference; this is not an independent study."""

    baseline: ParentTrial
    features: Features
    references: tuple[Calibration, ...]
    parent_ledger_sha256: str
    configurations: Literal[32] = 32
    candidates: Literal[8] = 8


class Sanity(Record):
    """The new pipeline must first reproduce the inherited strong baseline."""

    matches_original: bool
    parent_trial_id: str
    rows: tuple[Row, ...]


class Trial(Record):
    """Full start or terminal snapshot for one additional development hypothesis."""

    sequence: int
    trial_id: str
    batch_id: str
    batch_sha256: str
    config: Config
    features: Features
    hypothesis: str = Field(min_length=1)
    inputs: dict[str, str]
    environment: dict[str, str]
    started_at: str
    status: Literal["running", "keep", "discard", "crash"]
    rows: tuple[Row, ...] = ()
    score: float | None = None
    error: str | None = None


class Decision(Record):
    """A margin and harm decision on repeatedly used development data."""

    passed: bool
    reason: str
    winner: Config | None = None
    score: float | None = None
    delta: float | None = None
    negative_days: int | None = None
    inherited_configurations: Literal[32] = 32
    inherited_candidates: Literal[8] = 8
    additional_attempts: int


class Frozen(Record):
    """One pre-holdout selection retaining its complete inherited reference."""

    winner: Trial
    baseline: ParentTrial
    features: Features
    references: tuple[Calibration, ...]
    decision: Decision
    inputs: dict[str, str]
    batch_hashes: dict[str, str]
    ledger_sha256: str
    parent_ledger_sha256: str
    environment: dict[str, str]


def passes(trial: Trial, inherited: Inheritance) -> bool:
    """Require the fixed gain without adding harm beyond the inherited reference."""
    return bool(trial.rows) and (
        mean_r2(trial.rows) - mean_r2(inherited.baseline.rows) + EPSILON >= GATE_MARGIN
        and negative_days(trial.rows) <= negative_days(inherited.baseline.rows)
    )


def decide(events: tuple[Trial, ...], inherited: Inheritance) -> Decision:
    """Select the first passing follow-up while retaining inherited exposure counts."""
    successful = [event for event in events if event.score is not None]
    eligible = [event for event in successful if passes(event, inherited)]
    winner = (
        eligible[0]
        if eligible
        else max(successful, key=lambda event: mean_r2(event.rows), default=None)
    )
    return Decision(
        passed=bool(eligible),
        reason="Cumulative development gate passed"
        if eligible
        else "No additional candidate met the margin and harm rule on reused development days",
        additional_attempts=len({event.trial_id for event in events}),
        winner=winner.config if winner else None,
        score=mean_r2(winner.rows) if winner else None,
        delta=mean_r2(winner.rows) - mean_r2(inherited.baseline.rows) if winner else None,
        negative_days=negative_days(winner.rows) if winner else None,
    )
