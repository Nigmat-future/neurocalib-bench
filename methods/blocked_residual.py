"""Prefix-only blocked validation of a source-anchored residual update strength.

Three contiguous equal-row blocks are each predicted from the remaining prefix,
excluding twenty-five adjacent rows on either side. With the study's 20 ms bins,
this gap is at least 0.5 seconds. All ridge standardization is training-fold only.
The same ridge is then refit on the entire allowed prefix; no scoring data enter.

``config.alpha`` is the residual ridge penalty, initially fixed at ten.
``config.strength`` caps the selected update, initially one. ``bias_penalty``
is a dimensionless prior penalty on this scalar strength, centered at one half;
it does not penalize the residual intercept. Components and geometry stay fixed.
The scalar minimizes axis-weighted out-of-fold squared error plus
``bias_penalty * D * (strength - 0.5)**2``, where D is weighted correction energy.
Axis variances use prefix labels, floored at one percent of the source's global
velocity variance. This approximates macro-axis R2 without singular rank-one
weights. It is a prefix risk proxy, not an unbiased estimate of future risk.

Empty prefixes retain the original model. Fewer than 78 rows cannot form all
three purged folds and use the fixed half update, subject to the same cap.
"""

# ponytail: native namespace packaging needs no initialization file.
# ruff: noqa: INP001

from typing import Final

import numpy as np

from calibnext.models import Prepared
from calibnext.schema import Config
from neurocalib.core import Decoder, fit
from neurocalib.ledger import ResearchError
from neurocalib.schema import FloatArray

FOLDS: Final = 3
PURGE_ROWS: Final = 25
MIN_ROWS: Final = FOLDS * (PURGE_ROWS + 1)
VARIANCE_FLOOR: Final = 0.01
PRIOR_STRENGTH: Final = 0.5


def _out_of_fold(x: FloatArray, residual: FloatArray, alpha: float) -> FloatArray:
    """Predict each held block without its labels or its adjacent purge rows."""
    predictions = np.empty_like(residual)
    rows = np.arange(len(x))
    for held in np.array_split(rows, FOLDS):
        first, last = int(held[0]), int(held[-1]) + 1
        train = (rows < first - PURGE_ROWS) | (rows >= last + PURGE_ROWS)
        correction = fit(x[train], residual[train], alpha)
        predictions[held] = correction.predict(x[held])
    return predictions


def adapt(
    prepared: Prepared, prefix_x: FloatArray, prefix_y: FloatArray, config: Config
) -> Decoder:
    """Estimate one bounded prefix-only strength, then merge the full-prefix fit."""
    model = prepared.model
    n = len(prefix_x)
    if n == 0:
        return model
    residual = prefix_y - model.predict(prefix_x)
    strength = min(PRIOR_STRENGTH, config.strength)
    if n >= MIN_ROWS:
        if prepared.encoding is None:
            message = "blocked residual calibration requires the source velocity prior"
            raise ResearchError(message)
        source_variance = float(np.square(prepared.encoding.velocity_scale).sum() / 2)
        variance = np.maximum(prefix_y.var(axis=0), VARIANCE_FLOOR * source_variance)
        prediction = _out_of_fold(prefix_x, residual, config.alpha)
        energy = float(np.mean(np.sum(np.square(prediction) / variance, axis=1)))
        if energy > 0:
            cross = float(np.mean(np.sum(prediction * residual / variance, axis=1)))
            strength = float(
                np.clip(
                    (cross / energy + config.bias_penalty * PRIOR_STRENGTH)
                    / (1 + config.bias_penalty),
                    0,
                    config.strength,
                )
            )
    # ponytail: one fold-selected scalar keeps the full residual head and its fixed prior.
    correction = fit(prefix_x, residual, config.alpha)
    weights = model.scale[:, None] / correction.scale[:, None] * correction.weights
    bias = (
        correction.bias + (model.center - correction.center) / correction.scale @ correction.weights
    )
    return Decoder(
        model.center,
        model.scale,
        model.weights + strength * weights,
        model.bias + strength * bias,
    )
