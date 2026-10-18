"""Source-compatible LMMSE and three-predictor encoding calibration."""

import unittest

import numpy as np

from calibnext import encoding, models
from calibnext.schema import Config
from neurocalib.ledger import ResearchError


class EncodingChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(413)
        self.x = rng.normal(size=(900, 6)) + np.array([2, -1, 5, 4, -3, 1])
        self.y = self.x @ rng.normal(size=(6, 2)) + [3, -8]
        self.base = models.prepare(self.x, self.y, 0.3)
        self.source = encoding.prepare_source(
            self.base.model, self.base.covariance, len(self.y), self.y.T @ self.y, 0.3
        )
        self.rng = rng

    def test_source_noise_is_spd_and_bayes_readout_matches_original_ridge(self):
        # Given the regularized joint source covariance in fixed neural coordinates.
        source = self.source
        noise = np.linalg.solve(source.precision, np.eye(6))
        velocity = source.velocity_scale @ source.velocity_scale.T
        # When the source-compatible encoding factorization is reconstructed.
        expected = self.base.covariance + 0.3 * np.eye(6)
        # Then it is positive definite and reproduces the original discriminative readout.
        self.assertGreater(float(np.linalg.eigvalsh(noise).min()), 0)
        self.assertGreater(float(np.linalg.eigvalsh(velocity).min()), 0)
        np.testing.assert_allclose(
            noise + source.coefficients.T @ source.coefficients, expected, atol=1e-10
        )
        np.testing.assert_allclose(source.weights, self.base.model.weights, rtol=1e-10, atol=1e-10)

    def test_encoding_matches_penalized_three_predictor_equations(self):
        # Given arbitrary labeled prefix data and independently constructed encoding MAP equations.
        source = self.source
        x = self.rng.normal(size=(120, 6)) * [2, 3, 1, 4, 0.5, 2]
        y = x @ self.rng.normal(size=(6, 2)) + [5, -2]
        z = (x - source.model.center) / source.model.scale
        velocity = np.linalg.solve(source.velocity_scale, (y - source.model.bias).T).T
        design = np.column_stack([velocity, np.ones(len(x))])
        prior = np.vstack([source.coefficients, np.zeros(6)])
        penalty = np.diag([0.7, 0.7, 1.4])
        coefficients = prior + np.linalg.solve(
            design.T @ design / len(x) + penalty,
            design.T @ (z - design @ prior) / len(x),
        )
        neural = coefficients[:2]
        weights = (
            np.linalg.solve(
                (np.eye(2) + neural @ source.precision @ neural.T).T,
                neural @ source.precision,
            ).T
            @ source.velocity_scale.T
        )
        expected = source.model.weights + 0.4 * (weights - source.weights)
        # When source noise and velocity priors stay fixed during prefix adaptation.
        actual = encoding.adapt(
            source, x, y, Config(mode="encoding", alpha=0.7, strength=0.4, bias_penalty=1.4)
        )
        # Then weights and the separate neural-intercept correction match exactly.
        np.testing.assert_allclose(actual.weights, expected, atol=1e-10)
        np.testing.assert_allclose(
            actual.bias, source.model.bias - 0.4 * coefficients[-1] @ weights, atol=1e-10
        )
        np.testing.assert_array_equal(actual.center, source.model.center)
        np.testing.assert_array_equal(actual.scale, source.model.scale)

    def test_known_neural_encoding_change_is_recovered(self):
        # Given noiseless prefix observations from a known new encoding and offset.
        source = self.source
        velocity = self.rng.normal(size=(400, 2))
        neural = source.coefficients + self.rng.normal(size=(2, 6)) * 0.3
        offset = self.rng.normal(size=6)
        z = velocity @ neural + offset
        x = z * source.model.scale + source.model.center
        y = velocity @ source.velocity_scale.T + source.model.bias
        weighted = source.precision @ neural.T
        decoder = np.linalg.solve((np.eye(2) + neural @ weighted).T, weighted.T).T
        decoder = decoder @ source.velocity_scale.T
        # When the positive encoding prior approaches zero.
        actual = encoding.adapt(source, x, y, Config(mode="encoding", alpha=1e-10, bias_penalty=0))
        # Then the known generative model's LMMSE decoder, rather than an OLS decoder, is recovered.
        np.testing.assert_allclose(
            actual.weights, source.model.weights + decoder - source.weights, atol=1e-8
        )
        np.testing.assert_allclose(actual.bias, source.model.bias - offset @ decoder, atol=1e-8)

    def test_zero_prefix_or_zero_prior_innovation_preserves_source_exactly(self):
        # Given no prefix, or source-mean neural and behavioral observations.
        source = self.source
        x = np.tile(source.model.center, (20, 1))
        y = np.tile(source.model.bias, (20, 1))
        config = Config(mode="encoding")
        # When calibration sees no new evidence against the source encoding.
        empty = models.adapt(self.base, x[:0], y[:0], config)
        zero = encoding.adapt(source, x, y, config)
        # Then finite-precision source weights and offsets remain unchanged.
        self.assertIs(empty, self.base.model)
        np.testing.assert_array_equal(zero.weights, source.model.weights)
        np.testing.assert_array_equal(zero.bias, source.model.bias)

    def test_rank_deficient_velocity_prefix_stays_finite_and_source_priors_stay_fixed(self):
        # Given a prefix covering only one constant velocity with changing neural signals.
        source = self.source
        before = source.precision.copy()
        x = self.rng.normal(size=(50, 6))
        y = np.tile([4.0, -1.0], (50, 1))
        # When separate positive slope priors regularize this poorly covered prefix.
        actual = encoding.adapt(source, x, y, Config(mode="encoding", alpha=0.1, bias_penalty=0))
        # Then calibration is defined and never replaces source noise using target data.
        self.assertTrue(np.isfinite(actual.predict(x)).all())
        np.testing.assert_array_equal(source.precision, before)

    def test_missing_source_velocity_moments_cannot_silently_change_method(self):
        # Given old sufficient statistics that intentionally omit source behavioral covariance.
        old = models.prepare_moments(
            len(self.x), self.x.sum(0), self.y.sum(0), self.x.T @ self.x, self.x.T @ self.y, 0.3
        )
        # When encoding adaptation is requested without its declared source-only prior.
        with self.assertRaisesRegex(ResearchError, "velocity"):
            models.adapt(old, self.x[:30], self.y[:30], Config(mode="encoding"))
        # Then it must not silently fall back to MAP, target covariance, or another readout.


if __name__ == "__main__":
    unittest.main()
