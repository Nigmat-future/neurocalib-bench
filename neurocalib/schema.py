"""Validated experiment inputs and immutable numeric records."""

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type Mode = Literal["frozen", "center", "standardize", "coral", "supervised"]


class FileSpec(BaseModel):
    """A pinned public recording and its declared experimental role."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    name: str = Field(pattern=r"^indy_\d{8}_\d{2}\.mat$")
    role: Literal["source", "development", "test"]
    size: int = Field(gt=0)
    md5: str = Field(pattern=r"^[a-f0-9]{32}$")


class Protocol(BaseModel):
    """Read-only data and scoring contract; hash before opening final tests."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    version: str
    record_url: str
    bin_seconds: float = Field(gt=0, allow_inf_nan=False)
    score_start_seconds: float = Field(gt=30, allow_inf_nan=False)
    budgets_seconds: tuple[int, ...]
    files: tuple[FileSpec, ...]


class Strategy(BaseModel):
    """The only search surface: decoder and adaptation parameters."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    mode: Mode = "frozen"
    alpha: float = Field(default=1.0, gt=0, le=10000, allow_inf_nan=False)
    lags: int = Field(default=10, ge=1, le=20)
    strength: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    covariance_shrinkage: float = Field(default=0.1, gt=0, le=1, allow_inf_nan=False)


@dataclass(frozen=True, slots=True)
class Session:
    """One day's causal spike counts and backward-difference cursor velocity."""

    name: str
    times: FloatArray
    counts: FloatArray
    velocity: FloatArray
    channels: tuple[str, ...]
