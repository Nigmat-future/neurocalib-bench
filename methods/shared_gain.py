"""Unconstrained channel reweighting shared across neural history and both outputs.

The coefficients can be negative, so they do not represent pure firing-rate gains.
The objective divides joint two-axis squared error by the number of time bins,
then penalizes deviations from source channel weights and the normalized offset.
"""

# ponytail: native namespace packaging needs no initialization file.
# ruff: noqa: INP001

from typing import Final

import numpy as np

from calibnext.models import Prepared
from calibnext.schema import Config
from neurocalib.core import Decoder
from neurocalib.ledger import ResearchError
from neurocalib.schema import FloatArray

LAGS: Final = 10
AXES: Final = 2


def adapt(
    prepared: Prepared, prefix_x: FloatArray, prefix_y: FloatArray, config: Config
) -> Decoder:
    """Fit one channel coefficient and two offsets using only the observed prefix."""
    model = prepared.model
    n, dimensions = len(prefix_x), len(model.center)
    if not n:
        return model
    if dimensions == 0 or dimensions % LAGS:
        message = "shared channel reweighting requires complete 10-lag feature groups"
        raise ResearchError(message)
    if (
        prefix_x.shape != (n, dimensions)
        or prefix_y.shape != (n, AXES)
        or model.weights.shape != (dimensions, AXES)
    ):
        message = "prefix features and two-axis labels must match the source readout"
        raise ResearchError(message)
    if prepared.encoding is None:
        message = "shared channel reweighting requires the source velocity prior"
        raise ResearchError(message)
    rms = float(np.sqrt(np.square(prepared.encoding.velocity_scale).sum() / AXES))
    if not np.isfinite(rms) or rms <= 0:
        message = "source velocity RMS must be finite and positive"
        raise ResearchError(message)
    channels = dimensions // LAGS
    z = (prefix_x - model.center) / model.scale
    contributions = np.einsum(
        "nlc,lco->nco", z.reshape(n, LAGS, channels), model.weights.reshape(LAGS, channels, AXES)
    )
    design = np.empty((n * AXES, channels + AXES), dtype=np.float64)
    design[:, :channels] = contributions.transpose(0, 2, 1).reshape(n * AXES, channels) / rms
    design[:, channels:] = np.tile(np.eye(AXES), (n, 1))
    target = (prefix_y - model.predict(prefix_x)).reshape(-1) / rms
    penalty = np.diag(
        np.concatenate((np.full(channels, config.alpha), np.full(AXES, config.bias_penalty)))
    )
    theta = np.linalg.solve(design.T @ design / n + penalty, design.T @ target / n)
    coefficients = 1 + config.strength * theta[:channels]
    return Decoder(
        model.center,
        model.scale,
        (model.weights.reshape(LAGS, channels, AXES) * coefficients[None, :, None]).reshape(
            dimensions, AXES
        ),
        model.bias + config.strength * rms * theta[channels:],
    )
