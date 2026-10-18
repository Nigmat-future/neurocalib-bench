"""Shared causal neural preprocessing, selected using source days only."""

import numpy as np

from calib15.schema import Features
from neurocalib.core import Design, design
from neurocalib.schema import Session


def build(session: Session, channels: tuple[str, ...], features: Features, dt: float) -> Design:
    """Build past-only features, restarting smoothing at every recording gap."""
    if features.smoothing_seconds == 0:
        return design(session, channels, features.lags, dt)
    counts = session.counts.copy()
    alpha = -np.expm1(-dt / features.smoothing_seconds)
    continuous = np.isclose(np.diff(session.times), dt, atol=dt * 0.01)
    for index in range(1, len(counts)):
        if continuous[index - 1]:
            counts[index] = alpha * counts[index] + (1 - alpha) * counts[index - 1]
    smoothed = Session(session.name, session.times, counts, session.velocity, session.channels)
    return design(smoothed, channels, features.lags, dt)
