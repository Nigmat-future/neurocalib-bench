"""Causal features and small linear decoding baselines."""

from dataclasses import dataclass
from typing import Final, assert_never

import numpy as np

from neurocalib.schema import FloatArray, Session, Strategy

MIN_SCALE: Final = 1e-8


@dataclass(frozen=True, slots=True)
class Design:
    """Aligned causal features; time gaps and missing channels stay explicit."""

    x: FloatArray
    y: FloatArray
    times: FloatArray
    missing_channels: int


@dataclass(frozen=True, slots=True)
class Decoder:
    """An affine readout expressed in source-standardized coordinates."""

    center: FloatArray
    scale: FloatArray
    weights: FloatArray
    bias: FloatArray

    def predict(self, x: FloatArray) -> FloatArray:
        """Decode each row independently with no future observations."""
        return ((x - self.center) / self.scale) @ self.weights + self.bias


def design(session: Session, channels: tuple[str, ...], lags: int, dt: float) -> Design:
    """Stack current and past bins; drop every history crossing a time gap."""
    counts = np.zeros((len(session.times), len(channels)), dtype=np.float64)
    indices = {name: index for index, name in enumerate(session.channels)}
    for column, name in enumerate(channels):
        if name in indices:
            counts[:, column] = session.counts[:, indices[name]]
    if len(counts) < lags:
        msg = f"{session.name}: not enough bins for {lags} lags"
        raise ValueError(msg)
    windows = np.lib.stride_tricks.sliding_window_view(counts, lags, axis=0)
    x = windows[:, :, ::-1].transpose(0, 2, 1).reshape(len(windows), -1)
    time_windows = np.lib.stride_tricks.sliding_window_view(session.times, lags)
    valid = np.all(np.isclose(np.diff(time_windows, axis=1), dt, atol=dt * 0.01), axis=1)
    y = session.velocity[lags - 1 :]
    valid &= np.isfinite(x).all(axis=1)
    return Design(
        x[valid],
        y[valid],
        session.times[lags - 1 :][valid],
        sum(name not in indices for name in channels),
    )


def fit(x: FloatArray, y: FloatArray, alpha: float) -> Decoder:
    """Solve mean squared error plus alpha times squared readout norm."""
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > MIN_SCALE, scale, 1.0)
    z = (x - center) / scale
    bias = y.mean(axis=0)
    n, dimensions = len(z), len(center)
    # ponytail: use the smaller exact ridge system; no ML framework is needed.
    if n < dimensions:
        weights = z.T @ np.linalg.solve(z @ z.T / n + alpha * np.eye(n), (y - bias) / n)
    else:
        weights = np.linalg.solve(z.T @ z / n + alpha * np.eye(dimensions), z.T @ (y - bias) / n)
    return Decoder(center, scale, weights, bias)


def _covariance_power(x: FloatArray, shrinkage: float, power: float) -> FloatArray:
    """Regularize covariance before a symmetric square root or inverse root."""
    centered = x - x.mean(axis=0)
    covariance = centered.T @ centered / max(len(x) - 1, 1)
    covariance = (1 - shrinkage) * covariance + shrinkage * np.eye(x.shape[1])
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * np.maximum(values, 1e-10) ** power) @ vectors.T


def calibrate(
    model: Decoder,
    source_x: FloatArray,
    prefix_x: FloatArray,
    prefix_y: FloatArray | None,
    strategy: Strategy,
) -> Decoder:
    """Update from the allowed target prefix only; unlabeled modes need no y."""
    if len(prefix_x) == 0:
        return model
    z = (prefix_x - model.center) / model.scale
    mean = z.mean(axis=0)
    strength = strategy.strength
    match strategy.mode:
        case "frozen":
            return model
        case "center":
            weights = model.weights
            bias = model.bias - strength * (mean @ weights)
        case "standardize":
            scale = np.sqrt(
                (1 - strategy.covariance_shrinkage) * z.var(axis=0) + strategy.covariance_shrinkage
            )
            weights = ((1 - strength) + strength / scale)[:, None] * model.weights
            bias = model.bias - strength * ((mean / scale) @ model.weights)
        case "coral":
            source_z = (source_x - model.center) / model.scale
            transform = _covariance_power(
                z, strategy.covariance_shrinkage, -0.5
            ) @ _covariance_power(source_z, strategy.covariance_shrinkage, 0.5)
            weights = (1 - strength) * model.weights + strength * transform @ model.weights
            bias = model.bias - strength * (mean @ transform @ model.weights)
        case "supervised":
            if prefix_y is None:
                msg = "supervised calibration requires prefix behavior labels"
                raise ValueError(msg)
            residual = fit(prefix_x, prefix_y - model.predict(prefix_x), strategy.alpha)
            correction = (model.scale / residual.scale)[:, None] * residual.weights
            offset = ((model.center - residual.center) / residual.scale) @ residual.weights
            weights = model.weights + strength * correction
            bias = model.bias + strength * (residual.bias + offset)
        case unreachable:
            assert_never(unreachable)
    return Decoder(model.center, model.scale, weights, bias)


def metrics(truth: FloatArray, prediction: FloatArray) -> tuple[float, float]:
    """Return uniform-average two-axis R² and overall velocity RMSE (mm/s)."""
    residual = np.square(truth - prediction)
    variance = np.square(truth - truth.mean(axis=0)).sum(axis=0)
    if not np.all(variance > 0) or not np.isfinite(residual).all():
        msg = "non-finite predictions or constant evaluation target"
        raise ValueError(msg)
    return float(np.mean(1 - residual.sum(axis=0) / variance)), float(np.sqrt(residual.mean()))
