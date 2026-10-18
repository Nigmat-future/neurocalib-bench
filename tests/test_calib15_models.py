"""Synthetic recovery and coordinate invariants for prefix-only calibration."""

import unittest

import numpy as np

from calib15 import models
from calib15.schema import Calibration
from neurocalib.core import fit


class CalibrationChecks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(175)
        self.source_x = self.rng.normal(size=(400, 5)) * [2, 4, 0.5, 3, 1] + [4, -3, 20, 1, -2]
        self.source_y = self.source_x[:, :2] @ np.array([[2.0, -0.4], [0.6, 1.2]]) + [1, -7]
        self.prepared = models.prepare(self.source_x, self.source_y, 0.05)
        self.prefix = self.rng.normal(size=(120, 5)) * [3, 1, 5, 2, 0.2] + [-5, 3, 12, 8, 4]
        self.query = self.rng.normal(size=(80, 5)) * [2, 3, 1, 4, 2] + [2, 6, -3, 2, 8]

    def test_prepare_retains_source_only_orthonormal_basis(self):
        # Given five source coordinates and a request for only three components.
        original = self.source_x.copy()
        # When preparing the source representation.
        prepared = models.prepare(self.source_x, self.source_y, 0.05, max_components=3)
        # Then the basis is orthonormal and source input is unchanged.
        self.assertEqual(prepared.basis.shape, (5, 3))
        np.testing.assert_allclose(prepared.basis.T @ prepared.basis, np.eye(3), atol=1e-12)
        np.testing.assert_array_equal(self.source_x, original)

    def test_pca_uses_source_standardization_rather_than_raw_scale(self):
        # Given two correlated coordinates whose raw scales differ by fifty-fold.
        signs = np.tile([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], (30, 1))
        source = np.column_stack([100 * signs[:, 0] + 7, 2 * signs[:, 0] - 4, signs[:, 1]])
        # When retaining the single leading source principal direction.
        prepared = models.prepare(source, source[:, [0, 2]], 0.1, max_components=1)
        # Then its direction weights correlated standardized inputs equally, up to sign.
        expected = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0]])
        np.testing.assert_allclose(prepared.basis @ prepared.basis.T, expected, atol=1e-12)

    def test_zero_prefix_preserves_source_for_every_mode(self):
        # Given no observed target prefix for any comparison.
        for mode in ("frozen", "target", "residual", "affine", "subspace"):
            with self.subTest(mode=mode):
                # When calibration has zero observations.
                actual = models.adapt(
                    self.prepared, self.prefix[:0], self.source_y[:0], Calibration(mode=mode)
                )
                # Then the exact existing source model is returned.
                self.assertIs(actual, self.prepared.model)

    def test_target_is_an_independent_prefix_ridge(self):
        # Given a new mapping and a strength that must not blend target-only training.
        labels = self.prefix[:, 2:4] * [3, -2] + [9, 5]
        expected = fit(self.prefix, labels, 0.7).predict(self.query)
        # When fitting a standalone target decoder.
        actual = models.adapt(
            self.prepared, self.prefix, labels, Calibration(mode="target", alpha=0.7, strength=0.0)
        )
        # Then it matches independent ridge rather than a source mixture.
        np.testing.assert_allclose(actual.predict(self.query), expected, rtol=1e-12, atol=1e-12)

    def test_frozen_ignores_a_nonempty_prefix(self):
        # Given target observations whose labels disagree with the source model.
        labels = np.full((len(self.prefix), 2), 1e6)
        # When selecting the fixed-source comparison.
        actual = models.adapt(self.prepared, self.prefix, labels, Calibration(mode="frozen"))
        # Then no target data changes the source model.
        self.assertIs(actual, self.prepared.model)

    def test_full_residual_merges_prefix_center_and_scale_correctly(self):
        # Given source and target inputs with strongly different centers and scales.
        correction = self.prefix[:, 2:4] * [2, -4] + [3, 6]
        labels = self.prepared.model.predict(self.prefix) + correction
        residual = fit(self.prefix, correction, 0.2)
        expected = self.prepared.model.predict(self.query) + 0.4 * residual.predict(self.query)
        # When fitting a partial-strength full-dimensional residual.
        actual = models.adapt(
            self.prepared,
            self.prefix,
            labels,
            Calibration(mode="residual", alpha=0.2, strength=0.4),
        )
        # Then the merged affine decoder equals the unmerged two-stage prediction.
        np.testing.assert_allclose(actual.predict(self.query), expected, rtol=1e-11, atol=1e-11)

    def test_affine_recovers_a_known_output_transform(self):
        # Given an exact two-dimensional output rotation, gain change, and offset.
        transform = np.array([[1.3, -0.5], [0.2, 0.7]])
        offset = np.array([4.0, -8.0])
        labels = self.prepared.model.predict(self.prefix) @ transform + offset
        expected = self.prepared.model.predict(self.query) @ transform + offset
        # When calibrating only the output affine residual.
        actual = models.adapt(
            self.prepared, self.prefix, labels, Calibration(mode="affine", alpha=1e-10)
        )
        # Then query predictions recover that six-parameter mapping.
        np.testing.assert_allclose(actual.predict(self.query), expected, rtol=1e-8, atol=1e-8)

    def test_affine_strength_shrinks_the_update_toward_identity(self):
        # Given a regularized affine residual with a known independent solution.
        before = self.prepared.model.predict(self.prefix)
        labels = before * [1.4, 0.5] + [7, -2]
        head = fit(before, labels - before, 0.6)
        expected = self.prepared.model.predict(self.query)
        expected += 0.3 * head.predict(self.prepared.model.predict(self.query))
        # When applying thirty percent of that affine update.
        actual = models.adapt(
            self.prepared, self.prefix, labels, Calibration(mode="affine", alpha=0.6, strength=0.3)
        )
        # Then the closed-form merged prediction has the same regularization and strength.
        np.testing.assert_allclose(actual.predict(self.query), expected, rtol=1e-11, atol=1e-11)

    def test_subspace_recovers_a_known_fixed_source_basis_residual(self):
        # Given a target residual confined to two cached source principal directions.
        model = self.prepared.model
        basis = self.prepared.basis[:, :2]
        prefix_z = (self.prefix - model.center) / model.scale
        query_z = (self.query - model.center) / model.scale
        mapping = np.array([[0.8, -1.0], [1.4, 0.3]])
        labels = model.predict(self.prefix) + prefix_z @ basis @ mapping + [5, -4]
        expected = model.predict(self.query) + query_z @ basis @ mapping + [5, -4]
        # When adapting within that fixed source subspace.
        actual = models.adapt(
            self.prepared,
            self.prefix,
            labels,
            Calibration(mode="subspace", alpha=1e-10, components=2),
        )
        # Then query residuals are recovered without re-estimating target PCA.
        np.testing.assert_allclose(actual.predict(self.query), expected, rtol=1e-8, atol=1e-8)

    def test_subspace_limits_components_to_available_source_dimensions(self):
        # Given five source dimensions and the largest schema-allowed request.
        labels = self.prepared.model.predict(self.prefix) + self.prefix[:, :2]
        # When requesting thirty-two components.
        actual = models.adapt(
            self.prepared,
            self.prefix,
            labels,
            Calibration(mode="subspace", alpha=0.3, components=32),
        )
        # Then the available source basis is sufficient and all predictions are finite.
        self.assertEqual(self.prepared.basis.shape, (5, 5))
        self.assertTrue(np.isfinite(actual.predict(self.query)).all())

    def test_future_labels_cannot_change_prefix_calibration(self):
        # Given two recordings with the same prefix and different unseen suffix labels.
        full_x = np.vstack([self.prefix, self.query])
        full_y = full_x[:, :2] * [0.6, 2.0] + [5, -3]
        changed_y = full_y.copy()
        changed_y[len(self.prefix) :] += 1e6
        config = Calibration(mode="subspace", alpha=0.2, components=3)
        expected = models.adapt(self.prepared, self.prefix, full_y[:120], config)
        # When adapting from only the shared observed prefix.
        actual = models.adapt(self.prepared, full_x[:120], changed_y[:120], config)
        # Then future-label changes have no effect on either basis or predictions.
        np.testing.assert_array_equal(actual.predict(self.query), expected.predict(self.query))


if __name__ == "__main__":
    unittest.main()
