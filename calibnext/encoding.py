"""Source-compatible neural encoding updates with fixed source-only LMMSE priors."""

from dataclasses import dataclass
from typing import Final

import numpy as np

from calibnext.schema import Config
from neurocalib.core import Decoder
from neurocalib.schema import FloatArray

VELOCITY_JITTER: Final = 1e-6


@dataclass(frozen=True, slots=True)
class EncodingSource:
    """The original decoder, velocity whitening, encoding, and fixed noise precision."""

    model: Decoder
    velocity_scale: FloatArray
    coefficients: FloatArray
    precision: FloatArray
    weights: FloatArray


def _weights(
    coefficients: FloatArray, precision: FloatArray, velocity_scale: FloatArray
) -> FloatArray:
    """Convert a whitened-velocity encoding to its two-dimensional LMMSE readout."""
    weighted = precision @ coefficients.T
    normal = np.eye(len(coefficients)) + coefficients @ weighted
    return np.linalg.solve(normal.T, weighted.T).T @ velocity_scale.T


def prepare_source(
    model: Decoder,
    covariance: FloatArray,
    n: int,
    yy: FloatArray,
    source_alpha: float,
) -> EncodingSource:
    """Factor the existing ridge exactly, using only source behavioral moments."""
    velocity = yy / n - np.outer(model.bias, model.bias)
    velocity = (velocity + velocity.T) / 2
    jitter = VELOCITY_JITTER * max(float(np.trace(velocity)) / len(model.bias), 1.0)
    velocity_scale = np.linalg.cholesky(velocity + jitter * np.eye(len(model.bias)))
    gram = covariance + source_alpha * np.eye(len(model.center))
    cross = gram @ model.weights
    coefficients = np.linalg.solve(velocity_scale, cross.T)
    # The ridge noise floor makes this Schur complement positive definite.
    noise = gram - coefficients.T @ coefficients
    noise = (noise + noise.T) / 2
    precision = np.linalg.solve(noise, np.eye(len(noise)))
    precision = (precision + precision.T) / 2
    weights = _weights(coefficients, precision, velocity_scale)
    return EncodingSource(model, velocity_scale, coefficients, precision, weights)


def adapt(
    source: EncodingSource, prefix_x: FloatArray, prefix_y: FloatArray, config: Config
) -> Decoder:
    """Fit only a three-predictor prefix encoding; never update source covariance priors."""
    model = source.model
    if not len(prefix_x):
        return model
    z = (prefix_x - model.center) / model.scale
    velocity = np.linalg.solve(source.velocity_scale, (prefix_y - model.bias).T).T
    design = np.column_stack([velocity, np.ones(len(z))])
    prior = np.vstack([source.coefficients, np.zeros(len(model.center))])
    penalty = np.diag([config.alpha, config.alpha, config.bias_penalty])
    normal = design.T @ design / len(z) + penalty
    correction = np.linalg.solve(normal, design.T @ (z - design @ prior) / len(z))
    coefficients = prior + correction
    weights = _weights(coefficients[:-1], source.precision, source.velocity_scale)
    # ponytail: subtract the source Bayes readout to preserve the old ridge bit for bit.
    return Decoder(
        model.center,
        model.scale,
        model.weights + config.strength * (weights - source.weights),
        model.bias - config.strength * (coefficients[-1] @ weights),
    )
