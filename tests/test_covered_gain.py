"""A source-whitened prefix coverage guard preserves the exact shared-gain fit."""

import unittest

import numpy as np

from calibnext.models import Prepared, prepare
from calibnext.plugins import protect
from calibnext.schema import Config
from methods.covered_gain import adapt
from methods.shared_gain import adapt as shared_gain


class CoveredGainChecks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(290)
        source_x = self.rng.normal(size=(500, 30)) + np.arange(30) / 5
        source_y = source_x @ self.rng.normal(size=(30, 2)) * [30, 0.2] + [3, -7]
        self.prepared = prepare(source_x, source_y, 0.3)
        self.prefix = self.rng.normal(size=(80, 30)) * 2 - 1
        self.config = Config(
            mode="custom", method="covered_gain", alpha=0.01, bias_penalty=1, strength=1
        )

    def labels(self, white):
        encoding = self.prepared.encoding
        assert encoding is not None
        return white @ encoding.velocity_scale.T + self.prepared.model.bias

    def test_rank_one_constant_and_single_row_prefixes_preserve_the_source(self):
        # Given target motion restricted to one whitened direction, including rank-zero cases.
        for count in (1, 80):
            for amplitude in (0.0, 1.0):
                with self.subTest(prefix_rows=count, amplitude=amplitude):
                    trace = np.arange(count) * amplitude
                    labels = self.labels(np.column_stack((trace, -trace * 2)))
                    # When coverage lacks a second independent motion direction.
                    actual = adapt(self.prepared, self.prefix[:count], labels, self.config)
                    # Then no gain or intercept correction is applied.
                    self.assertIs(actual, self.prepared.model)

    def test_full_source_whitened_coverage_exactly_matches_shared_gain(self):
        # Given balanced whitened motion and highly unequal physical source axis scales.
        white = np.tile(np.array([[1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0]]), (20, 1))
        labels = self.labels(white)
        for bias_penalty in (0.0, 1.0):
            with self.subTest(bias_penalty=bias_penalty):
                config = self.config.model_copy(update={"bias_penalty": bias_penalty})
                # When the coverage condition passes despite poor unwhitened conditioning.
                expected = shared_gain(self.prepared, self.prefix, labels, config)
                actual = adapt(self.prepared, self.prefix, labels, config)
                # Then the existing gain objective is retained without approximation.
                np.testing.assert_array_equal(actual.weights, expected.weights)
                np.testing.assert_array_equal(actual.bias, expected.bias)
                self.assertFalse(np.array_equal(actual.weights, self.prepared.model.weights))

    def test_guarded_and_adapted_paths_leave_readonly_source_and_prefix_unchanged(self):
        # Given protected source arrays, neural observations and a full-coverage label prefix.
        protect(self.prepared)
        white = np.tile(np.array([[1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0]]), (20, 1))
        labels = self.labels(white)
        before_x, before_y = self.prefix.copy(), labels.copy()
        source = self.prepared.model
        before_weights, before_bias = source.weights.copy(), source.bias.copy()
        self.prefix.setflags(write=False)
        labels.setflags(write=False)
        # When either coverage path receives immutable calibration inputs.
        adapted = adapt(self.prepared, self.prefix, labels, self.config)
        frozen = adapt(self.prepared, self.prefix[:1], labels[:1], self.config)
        # Then both paths preserve every shared observation and source readout.
        self.assertIs(frozen, source)
        self.assertTrue(np.isfinite(adapted.weights).all())
        np.testing.assert_array_equal(self.prefix, before_x)
        np.testing.assert_array_equal(labels, before_y)
        np.testing.assert_array_equal(source.weights, before_weights)
        np.testing.assert_array_equal(source.bias, before_bias)

    def test_empty_prefix_preserves_source_without_optional_velocity_prior(self):
        # Given no target observations and no optional source velocity moments.
        prepared = self.prepared
        source = Prepared(prepared.model, prepared.covariance, prepared.basis)
        # When the zero-second calibration budget is requested.
        actual = adapt(source, self.prefix[:0], np.empty((0, 2)), self.config)
        # Then neither a coverage estimate nor a gain fit is attempted.
        self.assertIs(actual, prepared.model)


if __name__ == "__main__":
    unittest.main()
