"""Fixed data access, time budgets, and held-out scoring."""

from pathlib import Path
from time import perf_counter
from typing import Final, Literal

import numpy as np

from neurocalib.core import Design, calibrate, design, fit, metrics
from neurocalib.data import download_file, load_session
from neurocalib.schema import FloatArray, Protocol, Strategy

type ResultRow = dict[str, str | float | int]
MIN_SCORING_BINS: Final = 100


def read_protocol(root: Path) -> Protocol:
    """Parse the versioned protocol before reading recording data."""
    protocol = Protocol.model_validate_json((root / "protocol.json").read_text(encoding="utf-8"))
    if protocol.budgets_seconds != (0, 5, 15, 30):
        msg = "this evaluator requires the declared 0/5/15/30 second protocol"
        raise ValueError(msg)
    if protocol.score_start_seconds <= 30 + 20 * protocol.bin_seconds:
        msg = "scoring must start beyond the maximum prefix plus full history"
        raise ValueError(msg)
    return protocol


def source_design(root: Path, strategy: Strategy) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    """Choose channels from source days only; never join cross-day histories."""
    protocol = read_protocol(root)
    sessions = [
        load_session(download_file(spec, root / "data" / "raw"), protocol.bin_seconds)
        for spec in protocol.files
        if spec.role == "source"
    ]
    if not sessions:
        msg = "protocol needs source recordings"
        raise ValueError(msg)
    channels = tuple(
        name
        for name in sessions[0].channels
        if all(name in session.channels for session in sessions[1:])
    )
    if not channels:
        msg = "source recordings share no M1 channels"
        raise ValueError(msg)
    days = [design(session, channels, strategy.lags, protocol.bin_seconds) for session in sessions]
    x = np.concatenate([day.x[np.isfinite(day.y).all(axis=1)] for day in days])
    y = np.concatenate([day.y[np.isfinite(day.y).all(axis=1)] for day in days])
    return x, y, channels


def target_design(root: Path, name: str, channels: tuple[str, ...], strategy: Strategy) -> Design:
    """Build one day's features without selecting them using behavioral labels."""
    protocol = read_protocol(root)
    spec = next(item for item in protocol.files if item.name == name)
    return design(
        load_session(download_file(spec, root / "data" / "raw"), protocol.bin_seconds),
        channels,
        strategy.lags,
        protocol.bin_seconds,
    )


def evaluate(
    root: Path,
    strategy: Strategy,
    phase: Literal["development", "test"],
    *,
    shuffle_source: bool = False,
) -> list[ResultRow]:
    """Evaluate only the requested split under a shared scoring interval."""
    protocol = read_protocol(root)
    x, y, channels = source_design(root, strategy)
    if shuffle_source:
        y = np.roll(y, len(y) // 2, axis=0)
    model = fit(x, y, strategy.alpha)
    rows: list[ResultRow] = []
    for spec in protocol.files:
        if spec.role != phase:
            continue
        day = target_design(root, spec.name, channels, strategy)
        scoring = (day.times >= protocol.score_start_seconds) & np.isfinite(day.y).all(axis=1)
        if int(scoring.sum()) < MIN_SCORING_BINS:
            msg = f"{spec.name}: fewer than 100 scoring bins after the fixed cutoff"
            raise ValueError(msg)
        query, truth = day.x[scoring], day.y[scoring]
        for budget in protocol.budgets_seconds:
            prefix = day.times <= budget
            prefix_y = None
            if strategy.mode == "supervised":
                prefix &= np.isfinite(day.y).all(axis=1)
                prefix_y = day.y[prefix]
            start = perf_counter()
            adapted = calibrate(model, x, day.x[prefix], prefix_y, strategy)
            adaptation_time = perf_counter() - start
            predictions = adapted.predict(query)
            r2, rmse = metrics(truth, predictions)
            sse = np.square(truth - predictions).sum(axis=0)
            axis_r2 = 1 - sse / np.square(truth - truth.mean(axis=0)).sum(axis=0)
            # Measure actual single-bin calls, not amortized batch throughput.
            probe = query[: min(200, len(query))]
            start = perf_counter()
            for sample in probe:
                _ = adapted.predict(sample[None, :])
            inference_us = (perf_counter() - start) * 1e6 / len(probe)
            zero_r2, zero_rmse = metrics(truth, np.zeros_like(truth))
            rows.append(
                {
                    "session": spec.name,
                    "mode": strategy.mode,
                    "budget_seconds": budget,
                    "r2": r2,
                    "r2_x": float(axis_r2[0]),
                    "r2_y": float(axis_r2[1]),
                    "rmse": rmse,
                    "zero_r2": zero_r2,
                    "zero_rmse": zero_rmse,
                    "inference_us": inference_us,
                    "calibration_seconds": adaptation_time,
                    "missing_channels": day.missing_channels,
                    "n_scoring_bins": len(query),
                    "n_prefix_bins": int(prefix.sum()),
                    "n_channels": len(channels),
                    "n_features": query.shape[1],
                    "last_bin_seconds": float(day.times[-1]),
                }
            )
    if not rows:
        msg = f"no recordings for split {phase}"
        raise ValueError(msg)
    return rows
