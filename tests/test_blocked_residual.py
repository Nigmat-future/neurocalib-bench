"""Blocked shrinkage uses only isolated prefix fits and finite axis risk weights."""

import unittest

import numpy as np

from calibnext.models import Prepared, prepare
from calibnext.schema import Config
from methods.blocked_residual import _out_of_fold, adapt
from neurocalib.core import fit
from neurocalib.ledger import ResearchError


class BlockedResidualChecks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(719)
        source_x = self.rng.normal(size=(250, 4)) * [1, 3, 0.5, 2] + [7, -2, 3, 6]
        source_y = source_x @ self.rng.normal(size=(4, 2)) + [5, -9]
        self.prepared = prepare(source_x, source_y, 0.3)
        self.x = self.rng.normal(size=(150, 4)) + np.array([-3, 2, 6, -5])
        self.y = self.prepared.model.predict(self.x) + np.array([2, -3])
        self.y[:, 0] += np.repeat([-5, 0, 5], 50)
        self.query = self.rng.normal(size=(20, 4))

    def test_scalar_normal_equation_and_source_coordinate_merge(self):
        # Given three held blocks, independent source priors and an explicit scalar objective.
        source = self.prepared.model
        self.assertIsNotNone(self.prepared.encoding)
        assert self.prepared.encoding is not None
        residual = self.y - source.predict(self.x)
        predictions = []
        for first, last in ((0, 50), (50, 100), (100, 150)):
            rows = np.arange(len(self.x))
            train = (rows < first - 25) | (rows >= last + 25)
            readout = fit(self.x[train], residual[train], 10)
            predictions.append(readout.predict(self.x[first:last]))
        prediction = np.concatenate(predictions)
        source_variance = np.square(self.prepared.encoding.velocity_scale).sum() / 2
        variance = np.maximum(self.y.var(axis=0), 0.01 * source_variance)
        energy = np.mean(np.sum(prediction**2 / variance, axis=1))
        cross = np.mean(np.sum(prediction * residual / variance, axis=1))
        strength = np.clip((cross + 0.05 * energy) / (1.1 * energy), 0, 1)
        correction = fit(self.x, residual, 10)
        expected = source.predict(self.query) + strength * correction.predict(self.query)
        for array in (self.x, self.y, source.center, source.scale, source.weights, source.bias):
            array.setflags(write=False)
        # When the prior strength is one tenth and the final correction is capped.
        actual = adapt(
            self.prepared, self.x, self.y, Config(alpha=10, strength=1, bias_penalty=0.1)
        )
        # Then the analytic scalar optimum and source-coordinate predictions agree.
        np.testing.assert_allclose(actual.predict(self.query), expected, atol=1e-10, rtol=1e-10)

    def test_validation_and_purge_labels_never_enter_that_fold(self):
        # Given an isolated middle block with a twenty-five-row gap on both sides.
        residual = self.y - self.prepared.model.predict(self.x)
        reference = _out_of_fold(self.x, residual, 10)
        altered = residual.copy()
        altered[25:125] += [1000, -900]
        # When validation and purge labels change together.
        actual = _out_of_fold(self.x, altered, 10)
        # Then predictions for that held block are unchanged.
        np.testing.assert_array_equal(actual[50:100], reference[50:100])

    def test_rank_one_labels_have_finite_source_floored_risk(self):
        # Given a prefix with an exactly constant second velocity axis.
        labels = self.y.copy()
        labels[:, 1] = 7
        # When axis-normalized blocked calibration sees zero empirical variance.
        actual = adapt(self.prepared, self.x, labels, Config(bias_penalty=0))
        # Then neither the readout nor its predictions contain nonfinite values.
        self.assertTrue(np.isfinite(actual.predict(self.query)).all())
        self.assertTrue(np.isfinite(actual.weights).all())

    def test_empty_and_short_prefix_preserve_the_declared_fallback(self):
        # Given an absent velocity prior and too few rows for three purged folds.
        source = self.prepared
        prepared = Prepared(source.model, source.covariance, source.basis)
        # When calibration receives no observations or only thirty rows.
        empty = adapt(prepared, self.x[:0], self.y[:0], Config())
        actual = adapt(prepared, self.x[:30], self.y[:30], Config(bias_penalty=0))
        # Then the empty model is identical and the short prefix uses the fixed half update.
        self.assertIs(empty, source.model)
        correction = fit(self.x[:30], self.y[:30] - source.model.predict(self.x[:30]), 10)
        expected = source.model.predict(self.query) + 0.5 * correction.predict(self.query)
        np.testing.assert_allclose(actual.predict(self.query), expected, rtol=1e-10, atol=1e-10)

    def test_full_prefix_requires_source_variance_prior(self):
        # Given enough rows for blocked validation but no source velocity moments.
        source = self.prepared
        prepared = Prepared(source.model, source.covariance, source.basis)
        # When a source-floored validation loss cannot be defined.
        with self.assertRaisesRegex(ResearchError, "source velocity prior"):
            adapt(prepared, self.x, self.y, Config())
        # Then the missing scientific input is an explicit error.


if __name__ == "__main__":
    unittest.main()
