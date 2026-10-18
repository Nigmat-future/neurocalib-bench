"""Temporal residual penalties match dense ridge and retain the baseline limit."""

import unittest

import numpy as np

from calibnext.models import Prepared, prepare
from calibnext.schema import Config
from methods.temporal_residual import adapt
from neurocalib.core import MIN_SCALE, fit
from neurocalib.ledger import ResearchError


class TemporalResidualChecks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(721)
        source_x = self.rng.normal(size=(150, 30)) * np.arange(1, 31) + np.arange(30)
        source_y = source_x @ self.rng.normal(size=(30, 2)) + [4, -8]
        self.prepared = prepare(source_x, source_y, 0.3)
        self.prefix = self.rng.normal(size=(60, 30)) * np.arange(1, 31) / 3 - 4
        self.prefix[:, 0] = 3
        self.query = self.rng.normal(size=(20, 30)) * 2 + 5

    def test_dense_normal_equations_match_both_solve_branches_with_lag_major_order(self):
        # Given three channels ordered as ten consecutive lag blocks and protected inputs.
        source = self.prepared.model
        original_x, original_weights = self.prefix.copy(), source.weights.copy()
        for array in (self.prefix, source.center, source.scale, source.weights, source.bias):
            array.setflags(write=False)
        differences = np.diff(np.eye(10), axis=0)
        temporal = np.kron(differences, np.eye(3))
        for count in (8, 60):
            with self.subTest(prefix_rows=count):
                x = self.prefix[:count]
                labels = x[:, [2, 11]] * [1.2, -2.1] + [6, -9]
                labels.setflags(write=False)
                center = x.mean(0)
                scale = np.where(x.std(0) > MIN_SCALE, x.std(0), 1)
                z = (x - center) / scale
                design = np.column_stack((z, np.ones(count)))
                penalty = np.zeros((31, 31))
                penalty[:30, :30] = 10 * np.eye(30) + 7.0 * temporal.T @ temporal
                penalty[-1, -1] = 1.4
                # When the explicit standardized lag-difference objective is solved densely.
                coefficients = np.linalg.solve(
                    design.T @ design / count + penalty,
                    design.T @ (labels - source.predict(x)) / count,
                )
                expected = (
                    source.predict(self.query)
                    + 0.6
                    * np.column_stack(((self.query - center) / scale, np.ones(len(self.query))))
                    @ coefficients
                )
                actual = adapt(
                    self.prepared, x, labels, Config(alpha=7, strength=0.6, bias_penalty=1.4)
                )
                # Then the spectral solve preserves lag ordering, bias penalty and source merge.
                np.testing.assert_allclose(actual.predict(self.query), expected, atol=1e-10)
        np.testing.assert_array_equal(self.prefix, original_x)
        np.testing.assert_array_equal(source.weights, original_weights)

    def test_zero_temporal_penalty_limit_recovers_the_strong_baseline(self):
        # Given the fixed alpha-ten residual baseline, including an unpenalized offset.
        source = self.prepared.model
        for count in (8, 60):
            with self.subTest(prefix_rows=count):
                x = self.prefix[:count]
                labels = x[:, [2, 11]] * [1.2, -2.1] + [6, -9]
                reference = fit(x, labels - source.predict(x), 10)
                # When the temporal penalty approaches zero without changing neural shrinkage.
                actual = adapt(
                    self.prepared, x, labels, Config(alpha=1e-12, strength=0.5, bias_penalty=0)
                )
                # Then predictions match the original strong baseline to reproduction tolerance.
                expected = source.predict(self.query) + 0.5 * reference.predict(self.query)
                np.testing.assert_allclose(actual.predict(self.query), expected, atol=1e-7)

    def test_empty_prefix_preserves_the_exact_source_decoder(self):
        # Given zero allowed target observations and no optional source encoding.
        prepared = self.prepared
        source = Prepared(prepared.model, prepared.covariance, prepared.basis)
        # When temporal calibration receives the zero-second budget.
        actual = adapt(source, self.prefix[:0], np.empty((0, 2)), Config())
        # Then the exact source decoder is retained.
        self.assertIs(actual, prepared.model)

    def test_incomplete_lag_groups_are_rejected(self):
        # Given a source decoder whose feature count cannot represent ten lag blocks.
        source = prepare(self.prefix[:, :29], self.prefix[:, :2], 10)
        # When its features cross the temporal method boundary.
        with self.assertRaisesRegex(ResearchError, "10-lag"):
            adapt(source, self.prefix[:, :29], self.prefix[:, :2], Config())


if __name__ == "__main__":
    unittest.main()
