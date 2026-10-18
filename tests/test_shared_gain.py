"""Shared channel coefficients satisfy joint two-output ridge equations."""

import unittest

import numpy as np

from calibnext.models import Prepared, prepare
from calibnext.schema import Config
from methods.shared_gain import adapt
from neurocalib.ledger import ResearchError


class SharedChannelChecks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(281)
        self.source_x = self.rng.normal(size=(600, 30)) + np.arange(30) / 5
        source_y = self.source_x @ self.rng.normal(size=(30, 2)) + [3, -7]
        self.prepared = prepare(self.source_x, source_y, 0.3)
        self.prefix = self.rng.normal(size=(180, 30)) * 2 - 1
        self.query = self.rng.normal(size=(70, 30)) + 2

    def contributions(self, x):
        model = self.prepared.model
        z = (x - model.center) / model.scale
        return np.stack(
            [z[:, channel::3] @ model.weights[channel::3] for channel in range(3)], axis=1
        )

    def test_joint_normal_equations_and_global_source_scale_are_exact(self):
        # Given three channels, different source/prefix scales, and a penalized output offset.
        model = self.prepared.model
        original_x, original_w = self.prefix.copy(), model.weights.copy()
        residual = self.prefix[:, :2] * [1.3, -0.6] + [4, -2]
        labels = model.predict(self.prefix) + residual
        source = self.prepared.encoding
        self.assertIsNotNone(source)
        rms = np.sqrt(np.square(source.velocity_scale).sum() / 2)
        h = self.contributions(self.prefix)
        # When independent axis-major rows form the same joint optimization problem.
        design = np.zeros((2 * len(h), 5))
        design[: len(h), :3], design[len(h) :, :3] = h[:, :, 0] / rms, h[:, :, 1] / rms
        design[: len(h), 3], design[len(h) :, 4] = 1, 1
        target = np.concatenate([residual[:, 0], residual[:, 1]]) / rms
        theta = np.linalg.solve(
            design.T @ design / len(h) + np.diag([0.7, 0.7, 0.7, 1.4, 1.4]),
            design.T @ target / len(h),
        )
        actual = adapt(
            self.prepared, self.prefix, labels, Config(alpha=0.7, strength=0.6, bias_penalty=1.4)
        )
        correction = np.einsum("nco,c->no", self.contributions(self.query), theta[:3])
        expected = model.predict(self.query) + 0.6 * (correction + rms * theta[3:])
        # Then gain and intercept penalties match without changing input data or the source model.
        np.testing.assert_allclose(actual.predict(self.query), expected, rtol=1e-11, atol=1e-11)
        np.testing.assert_array_equal(self.prefix, original_x)
        np.testing.assert_array_equal(model.weights, original_w)
        np.testing.assert_array_equal(actual.center, model.center)
        np.testing.assert_array_equal(actual.scale, model.scale)

    def test_known_reweighting_is_shared_across_ten_lags_and_both_axes(self):
        # Given known channel weights including a negative value and a two-axis bias.
        model = self.prepared.model
        gains = np.array([0.65, 1.8, -0.4])
        offset = np.array([4.0, -3.0])
        labels = (
            model.bias + np.einsum("nco,c->no", self.contributions(self.prefix), gains) + offset
        )
        expected = (
            model.bias + np.einsum("nco,c->no", self.contributions(self.query), gains) + offset
        )
        # When essentially unregularized joint calibration recovers those coefficients.
        actual = adapt(self.prepared, self.prefix, labels, Config(alpha=1e-10, bias_penalty=0))
        # Then every lag and both output weights share each channel's coefficient, without clipping.
        wanted = model.weights.reshape(10, 3, 2) * gains[None, :, None]
        np.testing.assert_allclose(actual.weights.reshape(10, 3, 2), wanted, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(actual.bias, model.bias + offset, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(actual.predict(self.query), expected, rtol=1e-8, atol=1e-8)

    def test_zero_prefix_preserves_source_even_without_optional_velocity_prior(self):
        # Given no target rows and no optional encoding prior.
        prepared = Prepared(self.prepared.model, self.prepared.covariance, self.prepared.basis)
        # When no calibration information is available.
        actual = adapt(prepared, self.prefix[:0], np.empty((0, 2)), Config())
        # Then the exact original readout is returned.
        self.assertIs(actual, prepared.model)

    def test_missing_source_prior_or_incomplete_lag_groups_are_rejected(self):
        # Given missing source velocity moments, or a feature count not divisible by ten.
        old = Prepared(self.prepared.model, self.prepared.covariance, self.prepared.basis)
        labels = self.prepared.model.predict(self.prefix)
        # When calibration requests information absent from the fixed source representation.
        with self.assertRaisesRegex(ResearchError, "velocity"):
            adapt(old, self.prefix, labels, Config())
        source_y = self.prepared.model.predict(self.source_x)
        invalid = prepare(self.source_x[:, :-1], source_y, 0.3)
        with self.assertRaisesRegex(ResearchError, "10"):
            adapt(invalid, self.prefix[:, :-1], labels, Config())
        # Then no inferred lag layout or target normalization replaces the contract.


if __name__ == "__main__":
    unittest.main()
