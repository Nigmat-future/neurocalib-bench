"""Source-only preprocessing selection and budget-matched cross-day scoring."""

from __future__ import annotations

import json
from statistics import fmean
from time import perf_counter
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from calib15.features import build
from calib15.io import read_study, recording_path, results_dir
from calib15.models import Prepared, adapt, prepare
from calib15.schema import Calibration, Features, Recording, Row, Study
from neurocalib.core import Decoder, fit, metrics
from neurocalib.data import load_session
from neurocalib.ledger import ResearchError

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from neurocalib.core import Design
    from neurocalib.schema import FloatArray, Session

MIN_SCORING_BINS: Final = 100
PREDICTION_BATCH: Final = 2048
DIRECTIONS: Final = 8
MIN_DIRECTION_SPEED: Final = 5.0
SCORE_START_SECONDS: Final = 30.5


def _channels(sessions: tuple[Session, ...]) -> tuple[str, ...]:
    channels = tuple(
        name
        for name in sessions[0].channels
        if all(name in session.channels for session in sessions[1:])
    )
    if not channels:
        message = "same-animal source days have no common M1 channels"
        raise ValueError(message)
    return channels


def _sources(
    root: Path,
    study: Study,
    animal: str,
    features: Features,
) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    sessions = tuple(
        load_session(recording_path(root, spec), study.bin_seconds)
        for spec in study.files
        if spec.animal == animal and spec.role == "source"
    )
    if not sessions:
        message = f"no source days for {animal}"
        raise ValueError(message)
    channels = _channels(sessions)
    designs = [build(session, channels, features, study.bin_seconds) for session in sessions]
    source_x = np.concatenate([day.x[np.isfinite(day.y).all(axis=1)] for day in designs])
    source_y = np.concatenate([day.y[np.isfinite(day.y).all(axis=1)] for day in designs])
    return source_x, source_y, channels


def _predict(model: Decoder, x: FloatArray, indices: NDArray[np.int64]) -> FloatArray:
    # ponytail: bounded copies avoid duplicating a whole high-dimensional target day.
    return np.concatenate(
        [
            model.predict(x[indices[start : start + PREDICTION_BATCH]])
            for start in range(0, len(indices), PREDICTION_BATCH)
        ]
    )


def _score_day(
    spec: Recording,
    day: Design,
    prepared: Prepared,
    calibration: Calibration,
    budgets: tuple[int, ...],
) -> tuple[Row, ...]:
    scoring = (day.times >= SCORE_START_SECONDS) & np.isfinite(day.y).all(axis=1)
    indices = np.flatnonzero(scoring)
    if len(indices) < MIN_SCORING_BINS:
        message = f"{spec.name}: insufficient bins after the fixed 30.5 second cutoff"
        raise ValueError(message)
    truth = day.y[indices]
    frozen_r2, _ = metrics(truth, _predict(prepared.model, day.x, indices))
    rows: list[Row] = []
    for budget in budgets:
        prefix = (day.times <= budget) & np.isfinite(day.y).all(axis=1)
        prefix_x, prefix_y = day.x[prefix], day.y[prefix]
        start = perf_counter()
        model = adapt(prepared, prefix_x, prefix_y, calibration)
        calibration_seconds = perf_counter() - start
        prediction = _predict(model, day.x, indices)
        r2, rmse = metrics(truth, prediction)
        axis_r2 = 1 - np.square(truth - prediction).sum(axis=0) / np.square(
            truth - truth.mean(axis=0)
        ).sum(axis=0)
        speed = np.linalg.norm(prefix_y, axis=1)
        moving = prefix_y[speed >= MIN_DIRECTION_SPEED]
        angles = np.arctan2(moving[:, 1], moving[:, 0])
        occupied = np.floor((angles + np.pi) * DIRECTIONS / (2 * np.pi)).astype(int) % DIRECTIONS
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
                calibration_seconds=calibration_seconds,
                n_prefix_bins=len(prefix_x),
                n_scoring_bins=len(indices),
                n_features=day.x.shape[1],
                missing_channels=day.missing_channels,
                direction_bins=len(np.unique(occupied)),
                speed_std_mm_s=float(speed.std()) if len(speed) else 0.0,
            )
        )
    return tuple(rows)


def evaluate(
    root: Path,
    features: Features,
    calibration: Calibration,
    phase: Literal["development", "test"],
    budgets: tuple[int, ...] = (15,),
) -> tuple[Row, ...]:
    """Score one frozen feature set; holdout access is authorized before any data load."""
    if phase == "test":
        from calib15.records import read_frozen, require_final_access  # noqa: PLC0415

        require_final_access(root)
        frozen = read_frozen(root)
        study = read_study(root)
        if (
            features != frozen.features
            or calibration not in tuple(item.config for item in frozen.comparisons)
            or budgets != study.budgets_seconds
        ):
            message = (
                "test features, configuration and complete budgets must match the frozen comparison"
            )
            raise ResearchError(message)
    else:
        study = read_study(root)
    if (
        not budgets
        or len(set(budgets)) != len(budgets)
        or not set(budgets) <= set(study.budgets_seconds)
    ):
        message = "evaluation budgets must be distinct members of the fixed study budgets"
        raise ValueError(message)
    targets = tuple(spec for spec in study.files if spec.role == phase)
    if not targets:
        message = f"no target days for {phase}"
        raise ValueError(message)
    rows: list[Row] = []
    for animal in sorted({spec.animal for spec in targets}):
        source_x, source_y, channels = _sources(root, study, animal, features)
        prepared = prepare(source_x, source_y, features.source_alpha)
        for spec in targets:
            if spec.animal == animal:
                session = load_session(recording_path(root, spec), study.bin_seconds)
                day = build(session, channels, features, study.bin_seconds)
                rows.extend(
                    _score_day(
                        spec,
                        day,
                        prepared,
                        calibration,
                        budgets,
                    )
                )
                del day, session
        del prepared, source_x, source_y
    return tuple(rows)


def tune_source(root: Path) -> Features:
    """Select thirty fixed configurations by leave-one-Indy-source-day-out R²."""
    output = results_dir(root)
    if any(
        (output / name).exists()
        for name in (
            "frozen.json",
            "final.started.json",
            "experiments.jsonl",
        )
    ):
        message = "source tuning must finish before the first trial or final lock"
        raise ValueError(message)
    study = read_study(root)
    sessions = tuple(
        load_session(recording_path(root, spec), study.bin_seconds)
        for spec in study.files
        if spec.animal == "indy" and spec.role == "source"
    )
    if len(sessions) != len(("first", "second")):
        message = "source-only tuning requires exactly two Indy source days"
        raise ValueError(message)
    channels = _channels(sessions)
    trials: list[dict[str, float | list[str] | list[float] | dict[str, int | float]]] = []
    best_score = -float("inf")
    chosen = Features()
    for lags in (1, 5, 10):
        for smoothing in (0.0, 0.06):
            geometry = Features(lags=lags, smoothing_seconds=smoothing)
            days = [build(session, channels, geometry, study.bin_seconds) for session in sessions]
            for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
                candidate = geometry.model_copy(update={"source_alpha": alpha})
                scores: list[float] = []
                for index in range(len(days)):
                    train, target = days[index], days[1 - index]
                    valid = np.isfinite(train.y).all(axis=1)
                    model = fit(train.x[valid], train.y[valid], alpha)
                    scoring = (target.times >= study.score_start_seconds) & np.isfinite(
                        target.y
                    ).all(axis=1)
                    score, _ = metrics(target.y[scoring], model.predict(target.x[scoring]))
                    scores.append(score)
                average = fmean(scores)
                trials.append(
                    {
                        "features": candidate.model_dump(mode="json"),
                        "train_sessions": [session.name for session in sessions],
                        "score_sessions": [session.name for session in reversed(sessions)],
                        "scores": scores,
                        "mean_r2": average,
                    }
                )
                if average > best_score:
                    best_score, chosen = average, candidate
    output.mkdir(parents=True, exist_ok=True)
    _ = (output / "source_tuning.json").write_text(
        json.dumps(
            {
                "selection": "leave-one-Indy-source-day-out mean R2; fixed grid; first tie wins",
                "chosen": chosen.model_dump(mode="json"),
                "trials": trials,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _ = (output / "features.json").write_text(
        chosen.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return chosen
