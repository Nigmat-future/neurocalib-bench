"""Bounded-memory causal designs and exact source sufficient statistics."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from typing import Final

import numpy as np

from calib15.schema import Features
from neurocalib.core import Design, design
from neurocalib.schema import FloatArray, Session

BLOCK_SIZE: Final = 2048


@dataclass(frozen=True, slots=True)
class SourceStatistics:
    """Uncentered sums used to reconstruct the standardized source ridge fit."""

    n: int
    sum_x: FloatArray
    sum_y: FloatArray
    xx: FloatArray
    xy: FloatArray
    yy: FloatArray


def iter_design(
    session: Session,
    channels: tuple[str, ...],
    features: Features,
    dt: float,
    *,
    block_size: int = BLOCK_SIZE,
) -> Iterator[Design]:
    """Match the whole-day design exactly without a whole-day lag matrix."""
    if block_size < 1 or len(session.times) < features.lags:
        message = "positive block size and at least one complete history are required"
        raise ValueError(message)
    counts = session.counts
    if features.smoothing_seconds > 0:
        counts = session.counts.copy()
        alpha = -np.expm1(-dt / features.smoothing_seconds)
        continuous = np.isclose(np.diff(session.times), dt, atol=dt * 0.01)
        for index in range(1, len(counts)):
            if continuous[index - 1]:
                counts[index] = alpha * counts[index] + (1 - alpha) * counts[index - 1]
    # ponytail: keep the small channel matrix; only the expanded lag matrix is blocked.
    for start in range(features.lags - 1, len(counts), block_size):
        left, right = start - features.lags + 1, min(start + block_size, len(counts))
        block = Session(
            session.name,
            session.times[left:right],
            counts[left:right],
            session.velocity[left:right],
            session.channels,
        )
        yield design(block, channels, features.lags, dt)


def source_statistics(blocks: Iterable[Design]) -> SourceStatistics:
    """Accumulate finite-label source bins, keeping each day's histories separate."""
    iterator = iter(blocks)
    first = next(iterator, None)
    if first is None:
        message = "at least one source design block is required"
        raise ValueError(message)
    dimensions, outputs = first.x.shape[1], first.y.shape[1]
    n = 0
    sum_x, sum_y = np.zeros(dimensions), np.zeros(outputs)
    xx, xy = np.zeros((dimensions, dimensions)), np.zeros((dimensions, outputs))
    yy = np.zeros((outputs, outputs))
    for block in chain((first,), iterator):
        valid = np.isfinite(block.y).all(axis=1)
        x, y = block.x[valid], block.y[valid]
        n += len(x)
        sum_x += x.sum(axis=0)
        sum_y += y.sum(axis=0)
        xx += x.T @ x
        xy += x.T @ y
        yy += y.T @ y
    if n == 0:
        message = "source designs have no finite behavior labels"
        raise ValueError(message)
    return SourceStatistics(n, sum_x, sum_y, xx, xy, yy)
