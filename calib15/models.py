"""Closed-form supervised calibration in fixed source coordinates."""

from dataclasses import dataclass
from typing import Final, assert_never

import numpy as np

from calib15.schema import Calibration
from neurocalib.core import Decoder, fit
from neurocalib.schema import FloatArray

MAX_COMPONENTS: Final = 32


@dataclass(frozen=True, slots=True)
class Prepared:
    """A source decoder with its source-only principal directions cached."""

    model: Decoder
    source_x: FloatArray
    basis: FloatArray


def prepare(
    source_x: FloatArray,
    source_y: FloatArray,
    source_alpha: float,
    max_components: int = MAX_COMPONENTS,
) -> Prepared:
    """Fit the source readout and cache PCA of standardized source inputs."""
    if not 1 <= max_components <= MAX_COMPONENTS:
        message = "max_components must be between one and thirty-two"
        raise ValueError(message)
    model = fit(source_x, source_y, source_alpha)
    source_z = (source_x - model.center) / model.scale
    # ponytail: exact covariance PCA avoids an SVD of every source observation.
    _, vectors = np.linalg.eigh(source_z.T @ source_z / len(source_z))
    count = min(max_components, source_x.shape[1])
    return Prepared(model, source_x, vectors[:, -count:][:, ::-1])


def adapt(
    prepared: Prepared,
    prefix_x: FloatArray,
    prefix_y: FloatArray,
    config: Calibration,
) -> Decoder:
    """Fit only the observed prefix and merge its correction into one readout."""
    model = prepared.model
    if len(prefix_x) == 0:
        return model
    match config.mode:
        case "frozen":
            return model
        case "target":
            return fit(prefix_x, prefix_y, config.alpha)
        case "residual":
            head = fit(prefix_x, prefix_y - model.predict(prefix_x), config.alpha)
            correction = (model.scale / head.scale)[:, None] * head.weights
            offset = head.bias + ((model.center - head.center) / head.scale) @ head.weights
        case "affine":
            prediction = model.predict(prefix_x)
            head = fit(prediction, prefix_y - prediction, config.alpha)
            coefficients = head.weights / head.scale[:, None]
            correction = model.weights @ coefficients
            offset = head.bias + (model.bias - head.center) @ coefficients
        case "subspace":
            basis = prepared.basis[:, : config.components]
            prefix_z = (prefix_x - model.center) / model.scale
            head = fit(prefix_z @ basis, prefix_y - model.predict(prefix_x), config.alpha)
            coefficients = head.weights / head.scale[:, None]
            correction = basis @ coefficients
            offset = head.bias - head.center @ coefficients
        case unreachable:
            assert_never(unreachable)
    return Decoder(
        model.center,
        model.scale,
        model.weights + config.strength * correction,
        model.bias + config.strength * offset,
    )
