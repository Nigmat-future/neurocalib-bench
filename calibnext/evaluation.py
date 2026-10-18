"""Budget-identical evaluation with bounded neural feature memory."""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import TYPE_CHECKING, Literal, assert_never

import numpy as np

from calib15.evaluation import MIN_DIRECTION_SPEED, MIN_SCORING_BINS, SCORE_START_SECONDS
from calib15.features import build
from calib15.io import read_study, recording_path
from calib15.models import Prepared as OldPrepared
from calib15.models import adapt as old_adapt
from calib15.records import read_features
from calib15.schema import Calibration, Features, Recording, Row
from calibnext.data import iter_design, source_statistics
from calibnext.models import Prepared, adapt, prepare_moments
from calibnext.schema import Config
from neurocalib.core import Design, metrics
from neurocalib.data import load_session
from neurocalib.ledger import ResearchError
from neurocalib.schema import Session

if TYPE_CHECKING:
    from pathlib import Path

    from neurocalib.core import Decoder
    from neurocalib.schema import FloatArray


@lru_cache(maxsize=2)
def _prepare_animal(
    root: Path, animal: str, features_json: str
) -> tuple[Prepared, tuple[str, ...]]:
    features = Features.model_validate_json(features_json)
    study = read_study(root)
    sessions = tuple(
        load_session(recording_path(root, item), study.bin_seconds)
        for item in study.files
        if item.animal == animal and item.role == "source"
    )
    if not sessions:
        message = f"No source sessions for {animal}"
        raise ResearchError(message)
    channels = tuple(
        name
        for name in sessions[0].channels
        if all(name in session.channels for session in sessions[1:])
    )
    if not channels:
        message = f"No common source channels for {animal}"
        raise ResearchError(message)
    stats = source_statistics(
        block
        for session in sessions
        for block in iter_design(session, channels, features, study.bin_seconds)
    )
    prepared = prepare_moments(
        stats.n,
        stats.sum_x,
        stats.sum_y,
        stats.xx,
        stats.xy,
        features.source_alpha,
        yy=stats.yy,
    )
    return prepared, channels


def _prefix(
    session: Session, channels: tuple[str, ...], features: Features, budget: int, dt: float
) -> Design:
    selected = session.times <= budget
    if np.count_nonzero(selected) < features.lags:
        return Design(
            np.empty((0, len(channels) * features.lags)),
            np.empty((0, 2)),
            np.empty(0),
            sum(channel not in session.channels for channel in channels),
        )
    limited = Session(
        session.name,
        session.times[selected],
        session.counts[selected],
        session.velocity[selected],
        session.channels,
    )
    return build(limited, channels, features, dt)


def _score_day(  # noqa: PLR0913, PLR0917
    spec: Recording,
    session: Session,
    channels: tuple[str, ...],
    prepared: Prepared,
    config: Config | Calibration,
    budgets: tuple[int, ...],
    features: Features,
) -> tuple[Row, ...]:
    prefix = _prefix(session, channels, features, max(budgets), 0.02)
    models: list[Decoder] = []
    elapsed: list[float] = []
    prefix_labels: list[FloatArray] = []
    for budget in budgets:
        selected = (prefix.times <= budget) & np.isfinite(prefix.y).all(axis=1)
        x, y = prefix.x[selected], prefix.y[selected]
        start = perf_counter()
        match config:
            case Config():
                model = adapt(prepared, x, y, config)
            case Calibration():
                old = OldPrepared(
                    prepared.model,
                    np.empty((0, len(prepared.model.center))),
                    np.empty((len(prepared.model.center), 0)),
                )
                model = old_adapt(old, x, y, config)
            case unreachable:
                assert_never(unreachable)
        elapsed.append(perf_counter() - start)
        models.append(model)
        prefix_labels.append(y)
    predictions: list[list[FloatArray]] = [[] for _ in budgets]
    truth: list[FloatArray] = []
    frozen: list[FloatArray] = []
    for block in iter_design(session, channels, features, 0.02):
        selected = (block.times >= SCORE_START_SECONDS) & np.isfinite(block.y).all(axis=1)
        if not np.any(selected):
            continue
        x = block.x[selected]
        truth.append(block.y[selected])
        frozen.append(prepared.model.predict(x))
        for index, model in enumerate(models):
            predictions[index].append(model.predict(x))
    target = np.concatenate(truth) if truth else np.empty((0, 2))
    if len(target) < MIN_SCORING_BINS:
        message = f"{spec.name}: insufficient scoring bins"
        raise ResearchError(message)
    frozen_r2, _ = metrics(target, np.concatenate(frozen))
    rows: list[Row] = []
    for index, budget in enumerate(budgets):
        prediction = np.concatenate(predictions[index])
        r2, rmse = metrics(target, prediction)
        axis_r2 = 1 - np.square(target - prediction).sum(axis=0) / np.square(
            target - target.mean(axis=0)
        ).sum(axis=0)
        labels = prefix_labels[index]
        speed = np.linalg.norm(labels, axis=1)
        moving = labels[speed >= MIN_DIRECTION_SPEED]
        angles = np.arctan2(moving[:, 1], moving[:, 0])
        occupied = np.floor((angles + np.pi) * 8 / (2 * np.pi)).astype(int) % 8
        rows.append(
            Row(
                animal=spec.animal,
                session=spec.name,
                budget_seconds=budget,
                r2=r2,
                r2_x=float(axis_r2[0]),
                r2_y=float(axis_r2[1]),
                rmse=rmse,
                frozen_r2=frozen_r2,
                calibration_seconds=elapsed[index],
                n_prefix_bins=len(labels),
                n_scoring_bins=len(target),
                n_features=len(prepared.model.center),
                missing_channels=prefix.missing_channels,
                direction_bins=len(np.unique(occupied)),
                speed_std_mm_s=float(speed.std()) if len(speed) else 0,
            )
        )
    return tuple(rows)


def evaluate(
    root: Path,
    config: Config,
    phase: Literal["development", "test"] = "development",
    budgets: tuple[int, ...] = (15,),
) -> tuple[Row, ...]:
    """Adapt using prefix labels only; verify the final seal before loading data."""
    if phase == "test":
        from calibnext.records import read_frozen, require_final_access  # noqa: PLC0415

        require_final_access(root)
        frozen = read_frozen(root)
        if config != frozen.winner.config or budgets != (0, 5, 15, 30):
            message = "Final configuration and budgets must match the frozen study"
            raise ResearchError(message)
    return _evaluate(root, config, phase, budgets)


def evaluate_reference(
    root: Path,
    config: Calibration,
    phase: Literal["development", "test"] = "development",
    budgets: tuple[int, ...] = (15,),
) -> tuple[Row, ...]:
    """Recompute inherited comparisons through the identical streaming pipeline."""
    if phase == "test":
        from calibnext.records import read_frozen, require_final_access  # noqa: PLC0415

        require_final_access(root)
        frozen = read_frozen(root)
        if config not in frozen.references or budgets != (0, 5, 15, 30):
            message = "Final reference and budgets must match the frozen study"
            raise ResearchError(message)
    return _evaluate(root, config, phase, budgets)


def _evaluate(
    root: Path,
    config: Config | Calibration,
    phase: Literal["development", "test"],
    budgets: tuple[int, ...],
) -> tuple[Row, ...]:
    if (phase == "development" and budgets != (15,)) or (
        not budgets or len(set(budgets)) != len(budgets) or not set(budgets) <= {0, 5, 15, 30}
    ):
        message = "Development uses exactly 15 seconds; final uses the four frozen budgets"
        raise ResearchError(message)
    study, features = read_study(root), read_features(root)
    targets = tuple(item for item in study.files if item.role == phase)
    rows: list[Row] = []
    for animal in sorted({item.animal for item in targets}):
        # Cached preparation is safe only after verifying the pinned source bytes again.
        for source in study.files:
            if source.animal == animal and source.role == "source":
                _ = recording_path(root, source)
        prepared, channels = _prepare_animal(root, animal, features.model_dump_json())
        for spec in targets:
            if spec.animal == animal:
                session = load_session(recording_path(root, spec), study.bin_seconds)
                rows.extend(
                    _score_day(spec, session, channels, prepared, config, budgets, features)
                )
    return tuple(rows)
