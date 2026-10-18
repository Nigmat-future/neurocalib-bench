"""Bounded-memory source SIMPLS and source-anchored generalized ridge updates."""

from dataclasses import dataclass
from typing import Final, assert_never

import numpy as np

from calibnext.encoding import EncodingSource, prepare_source
from calibnext.encoding import adapt as adapt_encoding
from calibnext.schema import Config
from neurocalib.core import MIN_SCALE, Decoder, fit
from neurocalib.ledger import ResearchError
from neurocalib.schema import FloatArray

MAX_COMPONENTS: Final = 32
TOLERANCE: Final = 1e-12
SOURCE_SHRINKAGE: Final = 0.1


@dataclass(frozen=True, slots=True)
class Prepared:
    """One source decoder and its source-only covariance and supervised basis."""

    model: Decoder
    covariance: FloatArray
    basis: FloatArray
    encoding: EncodingSource | None = None


def _simpls(covariance: FloatArray, cross: FloatArray) -> FloatArray:
    """Run regression SIMPLS deflation using only centered sufficient statistics."""
    remaining = cross.copy()
    threshold = TOLERANCE * max(float(np.linalg.norm(cross)), 1.0)
    rotations: list[FloatArray] = []
    loadings: list[FloatArray] = []
    for _ in range(min(MAX_COMPONENTS, len(covariance))):
        left, singular, _ = np.linalg.svd(remaining, full_matrices=False)
        if not len(singular) or singular[0] <= threshold:
            break
        rotation: FloatArray = left[:, 0]
        variance = float(rotation @ covariance @ rotation)
        if variance <= TOLERANCE:
            break
        rotation = rotation / np.sqrt(variance)
        loading: FloatArray = covariance @ rotation
        if loadings:
            previous = np.column_stack(loadings)
            loading -= previous @ (previous.T @ loading)
        norm = float(np.linalg.norm(loading))
        if norm <= TOLERANCE:
            break
        loading /= norm
        remaining -= np.outer(loading, loading @ remaining)
        rotations.append(rotation)
        loadings.append(loading)
    if not rotations:
        return np.empty((len(covariance), 0), dtype=np.float64)
    # ponytail: QR keeps the supervised span while giving ridge an orthonormal basis.
    basis, _ = np.linalg.qr(np.column_stack(rotations))
    return basis


# The streaming caller supplies these six statistics directly, without an extra wrapper.
def prepare_moments(  # noqa: PLR0913, PLR0917
    n: int,
    sum_x: FloatArray,
    sum_y: FloatArray,
    xx: FloatArray,
    xy: FloatArray,
    alpha: float,
    *,
    yy: FloatArray | None = None,
) -> Prepared:
    """Fit standardized source ridge and SIMPLS from raw first and second moments."""
    if n <= 0:
        message = "source preparation requires at least one labeled observation"
        raise ValueError(message)
    center, bias = sum_x / n, sum_y / n
    covariance = xx / n - np.outer(center, center)
    scale = np.sqrt(np.maximum(np.diag(covariance), 0))
    scale = np.where(scale > MIN_SCALE, scale, 1.0)
    covariance = covariance / np.outer(scale, scale)
    covariance = (covariance + covariance.T) / 2
    cross = (xy / n - np.outer(center, bias)) / scale[:, None]
    weights = np.linalg.solve(covariance + alpha * np.eye(len(center)), cross)
    model = Decoder(center, scale, weights, bias)
    encoding = prepare_source(model, covariance, n, yy, alpha) if yy is not None else None
    return Prepared(model, covariance, _simpls(covariance, cross), encoding)


def prepare(source_x: FloatArray, source_y: FloatArray, source_alpha: float) -> Prepared:
    """Use the same sufficient-statistic source fit as the streaming evaluator."""
    return prepare_moments(
        len(source_x),
        source_x.sum(axis=0),
        source_y.sum(axis=0),
        source_x.T @ source_x,
        source_x.T @ source_y,
        source_alpha,
        yy=source_y.T @ source_y,
    )


def _map_update(
    prepared: Prepared, z: FloatArray, residual: FloatArray, config: Config
) -> tuple[FloatArray, FloatArray]:
    """Solve the declared MSE objective with a separately penalized intercept."""
    dimensions = z.shape[1]
    identity = np.eye(dimensions)
    match config.geometry:
        case "identity":
            precision = identity
        case "source":
            precision = (prepared.covariance + SOURCE_SHRINKAGE * identity) / (1 + SOURCE_SHRINKAGE)
        case unreachable:
            assert_never(unreachable)
    normal = np.empty((dimensions + 1, dimensions + 1), dtype=np.float64)
    normal[:-1, :-1] = z.T @ z / len(z) + config.alpha * precision
    normal[:-1, -1] = normal[-1, :-1] = z.mean(axis=0)
    normal[-1, -1] = 1 + config.bias_penalty
    response = np.vstack([z.T @ residual / len(z), residual.mean(axis=0)])
    solution = np.linalg.solve(normal, response)
    return solution[:-1], solution[-1]


def adapt(
    prepared: Prepared, prefix_x: FloatArray, prefix_y: FloatArray, config: Config
) -> Decoder:
    """Use only the labeled prefix; preserve fixed source input coordinates."""
    model = prepared.model
    if not len(prefix_x):
        return model
    z = (prefix_x - model.center) / model.scale
    residual = prefix_y - model.predict(prefix_x)
    match config.mode:
        case "encoding":
            if prepared.encoding is None:
                message = "encoding calibration requires source velocity second moments"
                raise ResearchError(message)
            return adapt_encoding(prepared.encoding, prefix_x, prefix_y, config)
        case "map":
            correction, offset = _map_update(prepared, z, residual, config)
        case "pls":
            basis = prepared.basis[:, : config.components]
            head = fit(z @ basis, residual, config.alpha)
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
