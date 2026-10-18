"""Development-inspired motion coverage guard for shared channel reweighting.

A source-whitened velocity covariance condition number above 100 keeps the
source decoder. This heuristic uses only prefix labels; it is not evidence
that deficient coverage causes negative transfer. Channel coefficients remain
unconstrained and therefore do not represent purely positive physical gains.
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
MIN_COVERAGE_RATIO: Final = 0.01


def adapt(
    prepared: Prepared, prefix_x: FloatArray, prefix_y: FloatArray, config: Config
) -> Decoder:
    """Preserve the source for deficient motion coverage; otherwise fit shared gains."""
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
    centered_velocity = prefix_y - prefix_y.mean(axis=0)
    white = np.linalg.solve(prepared.encoding.velocity_scale, centered_velocity.T).T
    eigenvalues = np.linalg.eigvalsh(white.T @ white / n)
    if eigenvalues[-1] <= 0 or eigenvalues[0] < MIN_COVERAGE_RATIO * eigenvalues[-1]:
        return model
    # ponytail: duplicate the shared-gain fit so this single script binds every dependency.
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
