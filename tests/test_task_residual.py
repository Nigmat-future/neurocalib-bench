"""Joint neural/task residual penalties survive scaling and analytic merging."""

import unittest

import numpy as np

from calibnext.models import Prepared, prepare
from calibnext.schema import Config
from methods.task_residual import adapt
from neurocalib.core import MIN_SCALE, fit


class TaskResidualChecks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(612)
        source_x = self.rng.normal(size=(400, 8)) * np.arange(1, 9) + np.arange(8) * 3
        source_y = source_x @ self.rng.normal(size=(8, 2)) + [4, -8]
        self.prepared = prepare(source_x, source_y, 0.3)
        self.prefix = self.rng.normal(size=(100, 8)) * 2 - np.arange(8)
        self.prefix[:, 0] = 3
        self.query = self.rng.normal(size=(70, 8)) + np.arange(8)

    def test_independent_augmented_normal_equations_match_primal_and_dual(self):
        # Given differently scaled source/prefix data, a constant feature and two sample regimes.
        source = self.prepared.model
        before_x, before_weights = self.prefix.copy(), source.weights.copy()
        for array in (self.prefix, source.center, source.scale, source.weights, source.bias):
            array.setflags(write=False)
        for count in (5, 100):
            with self.subTest(prefix_rows=count):
                x = self.prefix[:count]
                labels = x[:, 2:4] * [1.4, -2] + [7, -3]
                labels.setflags(write=False)
                prediction = source.predict(x)
                residual = labels - prediction
                x_center, p_center = x.mean(0), prediction.mean(0)
                x_scale = np.where(x.std(0) > MIN_SCALE, x.std(0), 1)
                p_scale = np.where(prediction.std(0) > MIN_SCALE, prediction.std(0), 1)
                design = np.column_stack(
                    [(x - x_center) / x_scale, (prediction - p_center) / p_scale, np.ones(count)]
                )
                penalty = np.diag([10.0] * 8 + [0.7, 0.7, 1.4])
                # When the original, unscaled coefficient blocks use their explicit penalties.
                coefficients = np.linalg.solve(
                    design.T @ design / count + penalty, design.T @ residual / count
                )
                query_design = np.column_stack(
                    [
                        (self.query - x_center) / x_scale,
                        (source.predict(self.query) - p_center) / p_scale,
                        np.ones(len(self.query)),
                    ]
                )
                expected = source.predict(self.query) + 0.6 * query_design @ coefficients
                actual = adapt(
                    self.prepared, x, labels, Config(alpha=0.7, strength=0.6, bias_penalty=1.4)
                )
                # Then both exact solve branches preserve separate neural/task/bias penalties.
                np.testing.assert_allclose(
                    actual.predict(self.query), expected, rtol=1e-10, atol=1e-10
                )
        np.testing.assert_array_equal(self.prefix, before_x)
        np.testing.assert_array_equal(source.weights, before_weights)
        np.testing.assert_array_equal(actual.center, source.center)
        np.testing.assert_array_equal(actual.scale, source.scale)

    def test_infinite_task_penalty_recovers_the_fixed_strong_residual_baseline(self):
        # Given the existing full residual baseline with alpha ten and strength one half.
        source = self.prepared.model
        for count in (5, 100):
            with self.subTest(prefix_rows=count):
                x = self.prefix[:count]
                labels = x[:, 2:4] * [1.4, -2] + [7, -3]
                residual = fit(x, labels - source.predict(x), 10)
                expected = source.predict(self.query) + 0.5 * residual.predict(self.query)
                # When the extra task pathway is suppressed without changing the neural penalty.
                actual = adapt(
                    self.prepared, x, labels, Config(alpha=1e16, strength=0.5, bias_penalty=0)
                )
                # Then merged predictions reproduce the baseline to the declared tolerance.
                np.testing.assert_allclose(
                    actual.predict(self.query), expected, rtol=1e-7, atol=1e-7
                )

    def test_neural_residual_retains_information_outside_source_predictions(self):
        # Given orthogonal balanced features and source output depending only on the first two.
        signs = np.array(
            [
                [a, b, c, d]
                for a in (-1.0, 1.0)
                for b in (-1.0, 1.0)
                for c in (-1.0, 1.0)
                for d in (-1.0, 1.0)
            ]
        )
        prepared = prepare(signs, signs[:, :2] * [2, -3], 0.3)
        source = prepared.model
        labels = source.predict(signs) + signs[:, 2, None] * [11, -22]
        # When the joint model receives a component absent from its two source predictions.
        actual = adapt(prepared, signs, labels, Config(alpha=0.2, bias_penalty=0))
        # Then neural ridge retains that component with the exact one-eleventh shrinkage.
        expected = source.predict(signs) + signs[:, 2, None] * [1, -2]
        np.testing.assert_allclose(actual.predict(signs), expected, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(actual.weights[2], [1, -2], rtol=1e-11, atol=1e-11)

    def test_empty_prefix_returns_the_same_source_without_optional_encoding(self):
        # Given no target observations and no source velocity prior.
        source = self.prepared
        prepared = Prepared(source.model, source.covariance, source.basis)
        # When the joint residual candidate has no calibration budget.
        actual = adapt(prepared, self.prefix[:0], np.empty((0, 2)), Config())
        # Then it preserves the exact original decoder.
        self.assertIs(actual, source.model)


if __name__ == "__main__":
    unittest.main()
