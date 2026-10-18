#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.4", "h5py==3.16.0", "pydantic==2.14.1"]
# ///
# ponytail: One standalone script preserves its strict recording whitelist and diagnostic bookkeeping. # noqa: E501  # noqa: SIZE_OK
# ─── How to run ───
# 1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
# 2. Run: uv run --script studies/calibnext/diagnose.py --root .
#    Or use .venv/Scripts/python.exe studies/calibnext/diagnose.py --root .
# 3. Numerical self-check only: append --self-check.
# ──────────────────
"""Exploratory diagnostics on the eleven already observed Indy recordings only."""

# ponytail: keep this one-off diagnostic standalone instead of adding report modules.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
parser = argparse.ArgumentParser(description=__doc__)
_ = parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
_ = parser.add_argument("--self-check", action="store_true")
arguments = parser.parse_args()
ROOT: Final = Path(arguments.root).resolve()
SELF_CHECK: Final = bool(arguments.self_check)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from calib15.io import read_study, recording_path  # noqa: E402
from calib15.models import Prepared, adapt  # noqa: E402
from calib15.records import Trial  # noqa: E402
from calib15.schema import Features, Recording  # noqa: E402
from calibnext.data import iter_design, source_statistics  # noqa: E402
from neurocalib.core import MIN_SCALE, Decoder  # noqa: E402
from neurocalib.data import load_session  # noqa: E402

if TYPE_CHECKING:
    from neurocalib.schema import FloatArray

PREFIX_SECONDS: Final = 15
SCORE_SECONDS: Final = 30.5

ALLOWED: Final = (
    "indy_20160930_05.mat",
    "indy_20161005_06.mat",
    "indy_20161011_03.mat",
    "indy_20161014_04.mat",
    "indy_20161017_02.mat",
    "indy_20161024_03.mat",
    "indy_20161025_04.mat",
    "indy_20161026_03.mat",
    "indy_20161027_03.mat",
    "indy_20161206_02.mat",
    "indy_20161207_02.mat",
)


class Spectrum(TypedDict):
    """Rank and temporal dependence, not an inferential effective sample size."""

    n: int
    numerical_rank: int
    participation_rank: float
    entropy_rank: float
    lag1: float
    initial_positive_iat_max_2s: float
    heuristic_effective_n: float


class DayDiagnostic(TypedDict):
    """One already observed development day's post-hoc decomposition."""

    session: str
    prefix_neural: Spectrum
    prefix_velocity: Spectrum
    source_metric_full_correction_capture: dict[str, float]
    r2: dict[str, float]
    prefix_residual_mean: list[float]
    scoring_residual_mean: list[float]
    mean_residual_cosine: float
    score_bins: int


def capture(weights: FloatArray, basis: FloatArray, covariance: FloatArray) -> float:
    """Fraction of centered prediction energy representable by a fixed span."""
    total = float(np.sum(weights * (covariance @ weights)))
    coordinates = np.linalg.pinv(basis.T @ covariance @ basis, hermitian=True) @ (
        basis.T @ covariance @ weights
    )
    projection = basis @ coordinates
    return float(np.sum(projection * (covariance @ projection)) / total) if total > 0 else 0.0


def spectrum(x: FloatArray) -> Spectrum:
    """Describe centered prefix rank and initial-positive autocorrelation pairs."""
    centered = x - x.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = singular**2
    proportions = eigenvalues / eigenvalues.sum()
    positive = proportions[proportions > 0]
    energy = float(np.square(centered).sum())
    correlations = [
        float(np.sum(centered[:-lag] * centered[lag:]) / energy)
        for lag in range(1, min(101, len(x)))
    ]
    tau = 1.0
    for index in range(0, len(correlations) - 1, 2):
        pair = correlations[index] + correlations[index + 1]
        if pair <= 0:
            break
        tau += 2 * pair
    return Spectrum(
        n=len(x),
        numerical_rank=int(
            np.count_nonzero(singular > singular[0] * max(x.shape) * np.finfo(float).eps)
        ),
        participation_rank=float(1 / np.square(proportions).sum()),
        entropy_rank=float(np.exp(-np.sum(positive * np.log(positive)))),
        lag1=correlations[0],
        initial_positive_iat_max_2s=tau,
        heuristic_effective_n=len(x) / tau,
    )


def source_model(
    sources: tuple[Recording, ...],
    features: Features,
) -> tuple[Prepared, FloatArray, tuple[str, ...], int]:
    """Recover the original source fit through bounded-memory moment accumulation."""
    sessions = tuple(load_session(recording_path(ROOT, spec)) for spec in sources)
    channels = tuple(c for c in sessions[0].channels if all(c in s.channels for s in sessions[1:]))
    moments = source_statistics(
        chain.from_iterable(iter_design(s, channels, features, 0.02) for s in sessions)
    )
    center, bias = moments.sum_x / moments.n, moments.sum_y / moments.n
    scale = np.sqrt(np.maximum(np.diag(moments.xx) / moments.n - center**2, 0))
    scale = np.where(scale > MIN_SCALE, scale, 1)
    covariance = (moments.xx / moments.n - np.outer(center, center)) / np.outer(scale, scale)
    cross = (moments.xy / moments.n - np.outer(center, bias)) / scale[:, None]
    weights = np.linalg.solve(covariance + features.source_alpha * np.eye(len(center)), cross)
    _, vectors = np.linalg.eigh(covariance)
    model = Decoder(center, scale, weights, bias)
    # ponytail: old adapt uses only model/basis; discarded source X is never materialized.
    prepared = Prepared(model, np.empty((0, len(center))), vectors[:, -32:][:, ::-1])
    return prepared, covariance, channels, moments.n


def diagnose_day(
    spec: Recording,
    features: Features,
    source: tuple[Prepared, FloatArray, tuple[str, ...], int],
    best: dict[str, Trial],
) -> DayDiagnostic:
    """Replay existing winners and decompose the already selected full residual."""
    prepared, covariance, channels, _ = source
    session = load_session(recording_path(ROOT, spec))
    blocks = iter_design(session, channels, features, 0.02)
    first = next(blocks)
    prefix = (first.times <= PREFIX_SECONDS) & np.isfinite(first.y).all(axis=1)
    x, y = first.x[prefix], first.y[prefix]
    model = prepared.model
    residual_mean = (y - model.predict(x)).mean(axis=0)
    models = {mode: adapt(prepared, x, y, trial.config) for mode, trial in best.items()}
    models["frozen"] = model
    strength = best["residual"].config.strength
    errors = {name: np.zeros(2) for name in (*models, "intercept_only", "slope_only")}
    sum_y, square_y, sum_residual = np.zeros(2), np.zeros(2), np.zeros(2)
    n = 0
    for block in chain((first,), blocks):
        valid = (block.times >= SCORE_SECONDS) & np.isfinite(block.y).all(axis=1)
        truth, z = block.y[valid], (block.x[valid] - model.center) / model.scale
        predictions = {name: z @ fitted.weights + fitted.bias for name, fitted in models.items()}
        predictions["intercept_only"] = predictions["frozen"] + strength * residual_mean
        predictions["slope_only"] = predictions["residual"] - strength * residual_mean
        for name, prediction in predictions.items():
            errors[name] += np.square(truth - prediction).sum(axis=0)
        n += len(truth)
        sum_y += truth.sum(axis=0)
        square_y += np.square(truth).sum(axis=0)
        sum_residual += (truth - predictions["frozen"]).sum(axis=0)
    variance = square_y - sum_y**2 / n
    scores = {name: float(np.mean(1 - error / variance)) for name, error in errors.items()}
    for mode, trial in best.items():
        expected = next(row.r2 for row in trial.rows if row.session == spec.name)
        np.testing.assert_allclose(scores[mode], expected, rtol=0, atol=1e-9)
    delta = models["residual"].weights - model.weights
    affine_basis = np.linalg.qr(model.weights)[0]
    scoring_mean = sum_residual / n
    return DayDiagnostic(
        session=spec.name,
        prefix_neural=spectrum((x - model.center) / model.scale),
        prefix_velocity=spectrum(y),
        source_metric_full_correction_capture={
            "pca32": capture(delta, prepared.basis, covariance),
            "affine2": capture(delta, affine_basis, covariance),
        },
        r2=scores,
        prefix_residual_mean=residual_mean.tolist(),
        scoring_residual_mean=scoring_mean.tolist(),
        mean_residual_cosine=float(
            residual_mean
            @ scoring_mean
            / (np.linalg.norm(residual_mean) * np.linalg.norm(scoring_mean))
        ),
        score_bins=n,
    )


def main() -> None:
    """Write exploratory evidence outside all frozen study inputs and results."""
    np.testing.assert_allclose(capture(np.ones((2, 1)), np.eye(2)[:, :1], np.diag([1.0, 4.0])), 0.2)
    if SELF_CHECK:
        return
    study = read_study(ROOT)
    selected = tuple(
        s for s in study.files if s.animal == "indy" and s.role in ("source", "development")
    )
    if {s.name for s in selected} != set(ALLOWED):
        message = "diagnostic access whitelist differs from the eleven observed Indy recordings"
        raise ValueError(message)
    ledger_bytes = (ROOT / "studies/calib15/results/experiments.jsonl").read_bytes()
    events = [Trial.model_validate_json(line) for line in ledger_bytes.splitlines()]
    latest = {trial.trial_id: trial for trial in events}
    completed = [trial for trial in latest.values() if trial.score is not None]
    best = {
        mode: max(
            (t for t in completed if t.config.mode == mode), key=lambda t: float(t.score or -1e99)
        )
        for mode in ("residual", "affine", "subspace")
    }
    before = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in events[0].inputs
    }
    if before != events[0].inputs:
        message = "bound scientific inputs changed before diagnostics"
        raise ValueError(message)
    features_path = ROOT / "studies/calib15/results/features.json"
    features = Features.model_validate_json(features_path.read_bytes())
    sources = tuple(s for s in selected if s.role == "source")
    fitted_source = source_model(sources, features)
    prepared, covariance, channels, n = fitted_source
    source = {
        "bins": n,
        "features": len(covariance),
        "channels": len(channels),
        "pca32_neural_variance_fraction": float(
            np.trace(prepared.basis.T @ covariance @ prepared.basis) / np.trace(covariance)
        ),
        "pca32_decoder_weight_fraction": capture(
            prepared.model.weights, prepared.basis, np.eye(len(covariance))
        ),
        "pca32_centered_prediction_energy_fraction": capture(
            prepared.model.weights, prepared.basis, covariance
        ),
    }
    print(json.dumps({"source": source}), flush=True)  # noqa: T201
    days: list[DayDiagnostic] = []
    for spec in selected:
        if spec.role == "development":
            row = diagnose_day(spec, features, fitted_source, best)
            days.append(row)
            print(json.dumps(row), flush=True)  # noqa: T201
    after = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in before}
    np.testing.assert_equal(before, after)
    output = {
        "exploratory_only": True,
        "recordings_opened": [s.name for s in selected],
        "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "bound_inputs": before,
        "bound_inputs_unchanged": before == after,
        "source": source,
        "days": days,
        "configs": {mode: trial.config.model_dump(mode="json") for mode, trial in best.items()},
        "development_exposures": {
            "observed_source_days": 2,
            "observed_development_days": 9,
            "existing_config_replay_count": 3,
            "frozen_reference_count": 1,
            "new_posthoc_config_count": 2,
            "new_posthoc_config_day_evaluations": 18,
            "first_execution": ".tools/diagnose_calib15.py; this script replays the same choices",
            "prefix_seconds": PREFIX_SECONDS,
            "score_start_seconds": SCORE_SECONDS,
            "posthoc_configs": [
                {
                    "name": "intercept_only",
                    "strength": best["residual"].config.strength,
                    "formula": "source_prediction + strength * mean(prefix_y-source_prediction)",
                },
                {
                    "name": "slope_only",
                    "strength": best["residual"].config.strength,
                    "formula": (
                        "full_residual_prediction - strength * mean(prefix_y-source_prediction)"
                    ),
                },
            ],
            "new_preregistered_trials": 0,
            "development_behavior_reused": True,
        },
        "notes": [
            "All diagnostics use observed development labels; no final or Loco recording opened.",
            (
                "IAT uses initial positive adjacent autocorrelation pairs, capped at 2 s; "
                "nonstationarity prevents inferential effective-n claims."
            ),
            "Intercept/slope-only ablations are post-hoc diagnostics, not preregistered trials.",
            (
                "Correction capture uses the source-covariance inner product; "
                "source prediction energy is centered and excludes intercept."
            ),
        ],
    }
    output_path = ROOT / "studies/calibnext/results/diagnosis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = output_path.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
