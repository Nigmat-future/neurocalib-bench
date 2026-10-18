"""Validated inputs and finite outputs for the separate calibration study."""

from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    """Reject undeclared fields and non-finite scientific values."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False
    )


class Recording(Record):
    """One official recording, with an explicit animal and access role."""

    name: str = Field(pattern=r"^(indy|loco)_\d{8}_\d{2}\.mat$")
    animal: Literal["indy", "loco"]
    role: Literal["source", "development", "test"]
    size: int = Field(gt=0)
    md5: str = Field(pattern=r"^[a-f0-9]{32}$")

    @model_validator(mode="after")
    def matching_animal(self) -> Self:
        """Prevent treating identically named channels as cross-animal identities."""
        if not self.name.startswith(self.animal + "_"):
            message = "recording name and animal disagree"
            raise ValueError(message)
        return self


class Study(Record):
    """Pinned chronological splits, information budget, and development gate."""

    version: Literal["calib15-v1"] = "calib15-v1"
    record_url: Literal["https://zenodo.org/records/3854034"]
    bin_seconds: float = Field(default=0.02, ge=0.02, le=0.02)
    score_start_seconds: float = Field(default=30.5, ge=30.5, le=30.5)
    primary_budget_seconds: Literal[15] = 15
    budgets_seconds: tuple[Literal[0, 5, 15, 30], ...] = (0, 5, 15, 30)
    max_candidates: Literal[12] = 12
    gate_delta: float = Field(default=0.02, ge=0.02, le=0.02)
    files: tuple[Recording, ...]

    @model_validator(mode="after")
    def distinct_dates(self) -> Self:
        """Require unique animal dates and targets strictly after their sources."""
        identities = {(item.animal, item.name.split("_")[1]) for item in self.files}
        if len(identities) != len(self.files) or self.budgets_seconds != (0, 5, 15, 30):
            message = "recordings must use distinct animal dates and all four fixed budgets"
            raise ValueError(message)
        for animal in {item.animal for item in self.files}:
            sources = [
                item.name for item in self.files if item.animal == animal and item.role == "source"
            ]
            targets = [
                item.name for item in self.files if item.animal == animal and item.role != "source"
            ]
            if not sources or not targets or max(sources) >= min(targets):
                message = "each animal needs sources strictly before its target dates"
                raise ValueError(message)
        if any(item.animal != "indy" for item in self.files if item.role == "development"):
            message = "only Indy development dates may select methods"
            raise ValueError(message)
        return self


class Features(Record):
    """Shared source-selected causal preprocessing and source regularization."""

    lags: int = Field(default=10, ge=1, le=20)
    smoothing_seconds: float = Field(default=0.0, ge=0, le=0.2)
    source_alpha: float = Field(default=1.0, gt=0, le=10000)


class Calibration(Record):
    """Explicit target-label access shared by all supervised comparisons."""

    mode: Literal["frozen", "target", "residual", "affine", "subspace"]
    alpha: float = Field(default=1.0, gt=0, le=10000)
    strength: float = Field(default=1.0, ge=0, le=1)
    components: int = Field(default=8, ge=1, le=32)


class Row(Record):
    """One animal, recording, and elapsed calibration budget."""

    animal: Literal["indy", "loco"]
    session: str
    budget_seconds: int = Field(ge=0)
    r2: float
    r2_x: float
    r2_y: float
    rmse: float = Field(ge=0)
    frozen_r2: float
    calibration_seconds: float = Field(ge=0)
    n_prefix_bins: int = Field(ge=0)
    n_scoring_bins: int = Field(gt=0)
    n_features: int = Field(gt=0)
    missing_channels: int = Field(ge=0)
    direction_bins: int = Field(ge=0, le=8)
    speed_std_mm_s: float = Field(ge=0)
