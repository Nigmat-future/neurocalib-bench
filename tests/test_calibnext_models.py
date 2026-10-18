"""Exact MAP equations and genuine supervised source subspaces on synthetic data."""

import unittest

import numpy as np

from calibnext import models
from calibnext.schema import Config
from neurocalib.core import fit


class ModelChecks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(514)
        mixing = np.eye(6) + np.tril(np.ones((6, 6)), -1) * 0.3
        self.source_x = self.rng.normal(size=(700, 6)) @ mixing + [4, -3, 20, 1, -2, 5]
        self.source_y = self.source_x @ self.rng.normal(size=(6, 2)) + [2, -6]
        self.prepared = models.prepare(self.source_x, self.source_y, 0.3)
        self.prefix = self.rng.normal(size=(120, 6)) * [3, 1, 5, 2, 0.2, 2] + [-5, 3, 12, 8, 4, 1]
        self.query = self.rng.normal(size=(80, 6)) * [2, 3, 1, 4, 2, 1] + [2, 6, -3, 2, 8, 4]

    def test_source_model_matches_existing_fit(self):
        # Given the same source observations and regularization.
        expected = fit(self.source_x, self.source_y, 0.3)
        # When sufficient statistics fit the new source decoder.
        actual = self.prepared.model
        # Then every prediction matches the existing two-pass standardized ridge.
        np.testing.assert_allclose(actual.center, expected.center, rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual.scale, expected.scale, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(
            actual.predict(self.query), expected.predict(self.query), rtol=1e-9, atol=1e-9
        )

    def test_chunked_sufficient_statistics_match_full_source_fit(self):
        # Given source moments accumulated independently across seven chunks.
        chunks = np.array_split(np.arange(len(self.source_x)), 7)
        sum_x = sum((self.source_x[index].sum(axis=0) for index in chunks), np.zeros(6))
        sum_y = sum((self.source_y[index].sum(axis=0) for index in chunks), np.zeros(2))
        xx = sum(
            (self.source_x[index].T @ self.source_x[index] for index in chunks), np.zeros((6, 6))
        )
        xy = sum(
            (self.source_x[index].T @ self.source_y[index] for index in chunks), np.zeros((6, 2))
        )
        # When preparing from only those bounded-memory statistics.
        actual = models.prepare_moments(len(self.source_x), sum_x, sum_y, xx, xy, 0.3)
        # Then source predictions and task subspaces match full-data preparation.
        np.testing.assert_allclose(
            actual.model.predict(self.query),
            self.prepared.model.predict(self.query),
            rtol=1e-9,
            atol=1e-9,
        )
        np.testing.assert_allclose(actual.covariance, self.prepared.covariance, atol=1e-11)
        np.testing.assert_allclose(
            actual.basis[:, :3] @ actual.basis[:, :3].T,
            self.prepared.basis[:, :3] @ self.prepared.basis[:, :3].T,
            atol=1e-9,
        )

    def test_map_matches_penalized_augmented_normal_equations(self):
        # Given differently centered target inputs, a nonzero bias penalty and both geometries.
        source = self.prepared.model
        z = (self.prefix - source.center) / source.scale
        residual = self.prefix[:, 2:4] * [1.5, -2] + [4, -3]
        labels = source.predict(self.prefix) + residual
        augmented = np.column_stack([z, np.ones(len(z))])
        for geometry in ("identity", "source"):
            with self.subTest(geometry=geometry):
                q = (
                    np.eye(6)
                    if geometry == "identity"
                    else (self.prepared.covariance + 0.1 * np.eye(6)) / 1.1
                )
                penalty = np.zeros((7, 7))
                penalty[:6, :6] = 0.7 * q
                penalty[-1, -1] = 1.5
                expected = np.linalg.solve(
                    augmented.T @ augmented / len(z) + penalty,
                    augmented.T @ residual / len(z),
                )
                # When adapting with partial update strength in fixed source coordinates.
                model = models.adapt(
                    self.prepared,
                    self.prefix,
                    labels,
                    Config(
                        mode="map", alpha=0.7, strength=0.6, geometry=geometry, bias_penalty=1.5
                    ),
                )
                # Then prediction equals the explicitly penalized least-squares solution.
                query_z = (self.query - source.center) / source.scale
                wanted = source.predict(self.query) + 0.6 * (query_z @ expected[:-1] + expected[-1])
                np.testing.assert_allclose(model.predict(self.query), wanted, atol=1e-10)
                np.testing.assert_array_equal(model.center, source.center)
                np.testing.assert_array_equal(model.scale, source.scale)

    def test_map_recovers_a_known_affine_neural_residual(self):
        # Given a full-dimensional residual and an unpenalized intercept.
        mapping = self.rng.normal(size=(6, 2))
        labels = self.prepared.model.predict(self.prefix) + self.prefix @ mapping + [5, -4]
        expected = self.prepared.model.predict(self.query) + self.query @ mapping + [5, -4]
        # When the positive ridge penalty approaches zero.
        model = models.adapt(
            self.prepared,
            self.prefix,
            labels,
            Config(mode="map", alpha=1e-10, bias_penalty=0),
        )
        # Then the exact residual mapping is recovered on unseen neural inputs.
        np.testing.assert_allclose(model.predict(self.query), expected, rtol=1e-7, atol=1e-7)

    def test_pls_basis_exceeds_two_output_dimensions_and_is_orthonormal(self):
        # Given six source dimensions, but only two behavioral outputs.
        z = (self.source_x - self.prepared.model.center) / self.prepared.model.scale
        cross = z.T @ (self.source_y - self.source_y.mean(axis=0))
        first, _, _ = np.linalg.svd(cross, full_matrices=False)
        # When true regression deflation builds the source task basis.
        basis = self.prepared.basis
        # Then components go beyond a rank-two cross-covariance SVD.
        self.assertEqual(basis.shape, (6, 6))
        np.testing.assert_allclose(basis.T @ basis, np.eye(6), atol=1e-12)
        np.testing.assert_allclose(
            basis[:, :1] @ basis[:, :1].T, first[:, :1] @ first[:, :1].T, atol=1e-12
        )

    def test_simpls_agrees_with_independent_pls1_data_deflation_subspace(self):
        # Given one behavioral response duplicated across both output coordinates.
        y = (self.source_x @ self.rng.normal(size=6))[:, None]
        prepared = models.prepare(self.source_x, np.hstack([y, 2 * y]), 0.3)
        x = (self.source_x - prepared.model.center) / prepared.model.scale
        response = (y - y.mean()).reshape(-1)
        weights, loadings = [], []
        # When a reference PLS1 algorithm explicitly deflates the observation matrix.
        for _ in range(3):
            weight = x.T @ response
            weight /= np.linalg.norm(weight)
            score = x @ weight
            loading = x.T @ score / (score @ score)
            coefficient = response @ score / (score @ score)
            weights.append(weight)
            loadings.append(loading)
            x -= np.outer(score, loading)
            response -= coefficient * score
        w, p = np.column_stack(weights), np.column_stack(loadings)
        reference, _ = np.linalg.qr(w @ np.linalg.solve(p.T @ w, np.eye(3)))
        # Then covariance-only SIMPLS finds the same supervised prediction subspace.
        np.testing.assert_allclose(
            prepared.basis[:, :3] @ prepared.basis[:, :3].T, reference @ reference.T, atol=1e-9
        )

    def test_pls_residual_merges_prefix_scaling_without_refitting_the_basis(self):
        # Given a target residual in two fixed source-supervised directions.
        source, basis = self.prepared.model, self.prepared.basis[:, :2]
        z = (self.prefix - source.center) / source.scale
        query_z = (self.query - source.center) / source.scale
        residual = z @ basis @ np.array([[0.8, -1.0], [1.4, 0.3]]) + [5, -4]
        labels = source.predict(self.prefix) + residual
        head = fit(z @ basis, residual, 0.2)
        expected = source.predict(self.query) + 0.4 * head.predict(query_z @ basis)
        original = self.prepared.basis.copy()
        # When fitting the prefix head and merging its own scale and intercept.
        actual = models.adapt(
            self.prepared,
            self.prefix,
            labels,
            Config(mode="pls", alpha=0.2, strength=0.4, components=2, bias_penalty=0),
        )
        # Then the closed-form readout matches the independent two-stage prediction.
        np.testing.assert_allclose(actual.predict(self.query), expected, atol=1e-10)
        np.testing.assert_array_equal(self.prepared.basis, original)

    def test_zero_prefix_keeps_source_for_both_modes(self):
        # Given no target-label observations.
        for mode in ("map", "pls"):
            # When either calibration family is selected.
            result = models.adapt(
                self.prepared, self.prefix[:0], self.source_y[:0], Config(mode=mode, bias_penalty=0)
            )
            # Then the exact existing source decoder survives.
            self.assertIs(result, self.prepared.model)

    def test_constant_source_behavior_gives_an_intercept_only_pls_update(self):
        # Given no source task covariance, so there is no supported supervised direction.
        prepared = models.prepare(self.source_x, np.ones_like(self.source_y), 0.3)
        labels = np.tile([5.0, -4.0], (len(self.prefix), 1))
        # When target calibration cannot use a source task component.
        actual = models.adapt(prepared, self.prefix, labels, Config(mode="pls", bias_penalty=0))
        # Then the empty basis is explicit and the unpenalized prefix intercept still works.
        self.assertEqual(prepared.basis.shape, (6, 0))
        np.testing.assert_allclose(actual.predict(self.query), np.tile([5.0, -4.0], (80, 1)))

    def test_empty_source_moments_are_rejected(self):
        # Given no valid labeled source rows.
        # When source fitting is requested from empty sufficient statistics.
        with self.assertRaisesRegex(ValueError, "at least one"):
            models.prepare_moments(
                0, np.zeros(6), np.zeros(2), np.zeros((6, 6)), np.zeros((6, 2)), 1
            )
        # Then there is no undefined source decoder or division by zero.


if __name__ == "__main__":
    unittest.main()
