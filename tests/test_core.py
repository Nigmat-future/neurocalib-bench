"""Scientific invariants for causal features and calibration."""

import unittest

import numpy as np

from neurocalib import core
from neurocalib.schema import Session, Strategy


class CoreChecks(unittest.TestCase):
    def test_history_uses_only_current_and_past_and_rejects_gaps(self):
        # Given counts with a missing time bin and one absent source channel.
        session = Session(
            "day",
            np.array([0.02, 0.04, 0.06, 0.10, 0.12]),
            np.arange(5.0, dtype=float)[:, None],
            np.ones((5, 2)),
            ("M1 001",),
        )
        # When constructing two-bin histories.
        design = core.design(session, ("M1 001", "M1 002"), 2, 0.02)
        # Then no history spans the gap; missing inputs are explicitly zero.
        np.testing.assert_allclose(design.times, [0.04, 0.06, 0.12])
        np.testing.assert_allclose(design.x, [[1, 0, 0, 0], [2, 0, 1, 0], [4, 0, 3, 0]])
        self.assertEqual(design.missing_channels, 1)

    def test_ridge_recovers_a_known_mapping_with_constant_input(self):
        # Given a linear process and a channel with zero variance.
        rng = np.random.default_rng(7)
        x = np.column_stack([rng.normal(size=(500, 3)), np.zeros(500)])
        y = x[:, :2] @ np.array([[2.0, -1.0], [0.5, 3.0]]) + 4
        # When training on the first block and predicting the last block.
        decoder = core.fit(x[:400], y[:400], 1e-6)
        prediction = decoder.predict(x[400:])
        # Then held-out behavior is recovered without singularity.
        self.assertGreater(core.metrics(y[400:], prediction)[0], 0.9999)

    def test_centering_recovers_mean_shift_using_prefix_inputs_only(self):
        # Given an exact map and a shifted target-day input distribution.
        rng = np.random.default_rng(12)
        source = rng.normal(size=(2000, 3))
        decoder = core.fit(source, source[:, :2], 1e-6)
        prefix = source + 10
        query = source[:200] + 10
        # When only unlabeled prefix inputs calibrate the decoder.
        adapted = core.calibrate(decoder, source, prefix, None, Strategy(mode="center"))
        # Then the offset is corrected without any behavior labels.
        self.assertGreater(core.metrics(source[:200, :2], adapted.predict(query))[0], 0.999)
        self.assertLess(core.metrics(source[:200, :2], decoder.predict(query))[0], 0)

    def test_zero_budget_preserves_the_frozen_decoder(self):
        # Given a fitted model and no target observations.
        x = np.arange(40.0, dtype=float).reshape(20, 2)
        decoder = core.fit(x, x, 1.0)
        # When a calibration strategy has zero data budget.
        adapted = core.calibrate(decoder, x, x[:0], None, Strategy(mode="coral"))
        # Then its predictions exactly equal the frozen model.
        np.testing.assert_array_equal(adapted.predict(x), decoder.predict(x))

    def test_covariance_alignment_improves_a_scaled_and_shifted_process(self):
        # Given a known affine distribution shift with no target labels.
        rng = np.random.default_rng(44)
        source = rng.normal(size=(4000, 3))
        model = core.fit(source, source[:, :2], 1e-6)
        target = source * [2.0, 0.5, 1.5] + 5
        # When regularized whitening/recoloring uses the target prefix only.
        adapted = core.calibrate(
            model, source, target, None, Strategy(mode="coral", covariance_shrinkage=0.01)
        )
        # Then held-out predictions improve over the frozen decoder.
        self.assertGreater(core.metrics(source[:200, :2], adapted.predict(target[:200]))[0], 0.97)

    def test_supervised_residual_corrects_slope_and_intercept(self):
        # Given a new day with changed output scale and intercept.
        rng = np.random.default_rng(25)
        x = rng.normal(size=(1000, 3))
        model = core.fit(x, x[:, :2], 1e-6)
        truth = 3 * x[:, :2] + 8
        # When target-prefix labels fit a regularized residual readout.
        adapted = core.calibrate(
            model, x, x[:200], truth[:200], Strategy(mode="supervised", alpha=1e-6)
        )
        # Then subsequent target predictions recover both slope and intercept.
        self.assertGreater(core.metrics(truth[200:], adapted.predict(x[200:]))[0], 0.9999)


if __name__ == "__main__":
    unittest.main()
