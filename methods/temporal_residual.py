"""Prefix residual ridge with a first-difference penalty across causal lags."""

# ponytail: native namespace packaging needs no initialization file.
# ruff: noqa: INP001

from typing import Final

import numpy as np

from calibnext.models import Prepared
from calibnext.schema import Config
from neurocalib.core import MIN_SCALE, Decoder
from neurocalib.ledger import ResearchError
from neurocalib.schema import FloatArray

LAGS: Final = 10
NEURAL_PENALTY: Final = 10.0


def adapt(
    prepared: Prepared, prefix_x: FloatArray, prefix_y: FloatArray, config: Config
) -> Decoder:
    """Fit mean residual loss plus neural and adjacent-lag coefficient penalties.

    Prefix-standardized coefficients B use penalty
    ``10 ||B||² + alpha ||(D_lag ⊗ I_channel) B||²``. The residual intercept
    has separate ``bias_penalty``. Zero temporal penalty recovers ordinary
    alpha-ten residual ridge; infinite temporal penalty retains constant-lag
    coefficient modes, independently for every channel and output axis.
    """
    model = prepared.model
    n, dimensions = prefix_x.shape
    if n == 0:
        return model
    if dimensions == 0 or dimensions % LAGS:
        message = "temporal residuals require complete 10-lag feature groups"
        raise ResearchError(message)
    center = prefix_x.mean(axis=0)
    scale = prefix_x.std(axis=0)
    scale = np.where(scale > MIN_SCALE, scale, 1.0)
    z = (prefix_x - center) / scale
    differences = np.diff(np.eye(LAGS), axis=0)
    eigenvalues, vectors = np.linalg.eigh(differences.T @ differences)
    inverse_scale = 1 / np.sqrt(NEURAL_PENALTY + config.alpha * np.maximum(eigenvalues, 0))
    transform = vectors * inverse_scale
    channels = dimensions // LAGS
    design: FloatArray = np.einsum("nlc,lk->nkc", z.reshape(n, LAGS, channels), transform).reshape(
        n, dimensions
    )
    residual = prefix_y - model.predict(prefix_x)
    residual_mean = residual.mean(axis=0)
    response = (residual - residual_mean) / n
    # ponytail: only ten lag eigenvectors are dense; the real 741-by-960 fit uses its dual.
    if n < dimensions:
        coefficients = design.T @ np.linalg.solve(design @ design.T / n + np.eye(n), response)
    else:
        coefficients = np.linalg.solve(
            design.T @ design / n + np.eye(dimensions), design.T @ response
        )
    standardized: FloatArray = np.einsum(
        "lk,kco->lco", transform, coefficients.reshape(LAGS, channels, 2)
    ).reshape(dimensions, 2)
    correction = standardized / scale[:, None]
    offset = residual_mean / (1 + config.bias_penalty) + (model.center - center) @ correction
    return Decoder(
        model.center,
        model.scale,
        model.weights + config.strength * model.scale[:, None] * correction,
        model.bias + config.strength * offset,
    )
