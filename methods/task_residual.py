"""Joint neural and source-prediction residuals with separate ridge penalties."""

# ponytail: native namespace packaging needs no initialization file.
# ruff: noqa: INP001

import numpy as np

from calibnext.models import Prepared
from calibnext.schema import Config
from neurocalib.core import MIN_SCALE, Decoder
from neurocalib.schema import FloatArray


def _standardize(x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Center one prefix block without erasing its later penalty scaling."""
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > MIN_SCALE, scale, 1.0)
    return (x - center) / scale, center, scale


def adapt(
    prepared: Prepared, prefix_x: FloatArray, prefix_y: FloatArray, config: Config
) -> Decoder:
    """Fit prefix-only residuals and merge both paths into source coordinates.

    Neural coefficients have fixed penalty ten. Source-prediction coefficients
    have penalty ``config.alpha``. Both blocks use prefix-standardized features;
    the residual intercept has independent penalty ``config.bias_penalty``.
    """
    model = prepared.model
    n, dimensions = prefix_x.shape
    if n == 0:
        return model
    prediction = model.predict(prefix_x)
    neural_z, neural_center, neural_scale = _standardize(prefix_x)
    task_z, task_center, task_scale = _standardize(prediction)
    neural_factor = np.sqrt(10.0)
    task_factor = np.sqrt(config.alpha)
    design = np.column_stack((neural_z / neural_factor, task_z / task_factor))
    residual = prefix_y - prediction
    residual_mean = residual.mean(axis=0)
    centered = residual - residual_mean
    # ponytail: solve the smaller exact system; do not standardize this design again.
    if n < design.shape[1]:
        coefficients = design.T @ np.linalg.solve(design @ design.T / n + np.eye(n), centered / n)
    else:
        coefficients = np.linalg.solve(
            design.T @ design / n + np.eye(design.shape[1]), design.T @ centered / n
        )
    neural = coefficients[:dimensions] / neural_factor / neural_scale[:, None]
    task = coefficients[dimensions:] / task_factor / task_scale[:, None]
    weights = model.scale[:, None] * neural + model.weights @ task
    bias = (
        residual_mean / (1.0 + config.bias_penalty)
        + (model.center - neural_center) @ neural
        + (model.bias - task_center) @ task
    )
    return Decoder(
        model.center,
        model.scale,
        model.weights + config.strength * weights,
        model.bias + config.strength * bias,
    )
