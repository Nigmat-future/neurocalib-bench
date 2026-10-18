"""Load trusted, single-file research methods from their actual source bytes."""

from __future__ import annotations

import ast
import sys
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Final, Protocol, assert_never, runtime_checkable

import numpy as np

from calibnext.models import Prepared, adapt
from calibnext.schema import Config
from neurocalib.core import Decoder
from neurocalib.ledger import ResearchError
from neurocalib.schema import FloatArray

if TYPE_CHECKING:
    from pathlib import Path

type Calibrator = Callable[[Prepared, FloatArray, FloatArray, Config], Decoder]
ALLOWED_IMPORTS: Final = sys.stdlib_module_names | {"numpy", "calibnext", "calib15", "neurocalib"}


@runtime_checkable
class CandidateModule(Protocol):
    """The one fixed research boundary: observed prefix in, linear decoder out."""

    def adapt(
        self, prepared: Prepared, prefix_x: FloatArray, prefix_y: FloatArray, config: Config
    ) -> Decoder:
        """Return a prefix-fitted readout without reading any other target observations."""
        ...


def load_method(root: Path, config: Config) -> Calibrator:
    """Compile fresh source; the import rule tracks dependencies, not a security sandbox."""
    match config.mode:
        case "map" | "pls" | "encoding":
            return adapt
        case "custom":
            directory = root / "methods"
            path = directory / f"{config.method}.py"
            if path.resolve().parent != directory.resolve() or not path.is_file():
                message = "Candidate method must resolve inside the methods directory"
                raise ResearchError(message)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                match node:
                    case ast.Import(names=names):
                        modules = [name.name.split(".")[0] for name in names]
                    case ast.ImportFrom(module=imported_module, level=level):
                        modules = (
                            [imported_module.split(".")[0]]
                            if imported_module is not None and not level
                            else [""]
                        )
                    case ast.AST():
                        continue
                    case unreachable:
                        assert_never(unreachable)
                if not set(modules) <= ALLOWED_IMPORTS:
                    message = (
                        "Candidates must be single-file methods using stdlib, NumPy or bound core"
                    )
                    raise ResearchError(message)
            name = f"_neurocalib_candidate_{config.method}"
            module = ModuleType(name)
            module.__file__ = str(path)
            sys.modules[name] = module
            # ponytail: compiling current bytes avoids timestamp-based stale pyc reuse.
            exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102
            if not isinstance(module, CandidateModule):
                message = "Candidate method must export adapt(prepared, prefix_x, prefix_y, config)"
                raise ResearchError(message)
            return module.adapt
        case unreachable:
            assert_never(unreachable)


def protect(prepared: Prepared) -> None:
    """Mark cached source arrays read-only before any candidate receives them."""
    model = prepared.model
    arrays = (
        model.center,
        model.scale,
        model.weights,
        model.bias,
        prepared.covariance,
        prepared.basis,
    )
    if prepared.encoding is not None:
        source = prepared.encoding
        arrays += (source.velocity_scale, source.coefficients, source.precision, source.weights)
    for array in arrays:
        array.setflags(write=False)


def calibrate(
    adapter: Calibrator,
    prepared: Prepared,
    prefix_x: FloatArray,
    prefix_y: FloatArray,
    config: Config,
) -> Decoder:
    """Protect observations and reject malformed or nonfinite candidate readouts."""
    prefix_x.setflags(write=False)
    prefix_y.setflags(write=False)
    model = adapter(prepared, prefix_x, prefix_y, config)
    dimensions = len(prepared.model.center)
    if (
        model.center.shape != (dimensions,)
        or model.scale.shape != (dimensions,)
        or model.weights.shape != (dimensions, 2)
        or model.bias.shape != (2,)
        or any(
            not np.isfinite(array).all()
            for array in (model.center, model.scale, model.weights, model.bias)
        )
        or not np.all(model.scale > 0)
    ):
        message = (
            "Candidate must return a finite, dimension-compatible decoder with positive scales"
        )
        raise ResearchError(message)
    return model
