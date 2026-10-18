"""Parity checks for bounded-memory causal designs and source moments."""

import unittest

import numpy as np

from calib15.features import build
from calib15.schema import Features
from calibnext.data import iter_design, source_statistics
from neurocalib.core import fit
from neurocalib.schema import Session


class TestStreamingData(unittest.TestCase):
    def test_design_matches_whole_day_when_gaps_ema_and_missing_channels(self):
        # Given a gap inside a smoothing history and a missing physical channel.
        rng = np.random.default_rng(41)
        times = np.arange(1, 240, dtype=np.float64) * np.float64(0.02)
        times[91:] += 0.1
        session = Session(
            "synthetic",
            times,
            rng.poisson(0.3, (len(times), 3)).astype(float),
            rng.normal(size=(len(times), 2)),
            ("M1_1", "M1_2", "M1_3"),
        )
        channels = ("M1_3", "missing", "M1_1")
        for lags in (1, 5, 10):
            for smoothing in (0.0, 0.06):
                features = Features(lags=lags, smoothing_seconds=smoothing)
                expected = build(session, channels, features, 0.02)
                # When the same session is designed in small overlapping blocks.
                blocks = tuple(iter_design(session, channels, features, 0.02, block_size=17))
                # Then every aligned value and time is identical, including gap rejection.
                np.testing.assert_array_equal(np.concatenate([day.x for day in blocks]), expected.x)
                np.testing.assert_array_equal(np.concatenate([day.y for day in blocks]), expected.y)
                np.testing.assert_array_equal(
                    np.concatenate([day.times for day in blocks]),
                    expected.times,
                )
                self.assertTrue(all(day.missing_channels == 1 for day in blocks))

    def test_moments_reconstruct_fit_when_labels_missing_and_source_days_separate(self):
        # Given two source sessions, one missing label and one constant feature.
        rng = np.random.default_rng(5)
        sessions: list[Session] = []
        for name in ("first", "second"):
            counts = rng.normal(size=(120, 3))
            counts[:, 1] = 7
            velocity = rng.normal(size=(120, 2))
            velocity[13] = np.nan
            times = np.arange(120, dtype=np.float64) * np.float64(0.02)
            sessions.append(Session(name, times, counts, velocity, ("a", "b", "c")))
        features = Features(lags=5)
        days = [build(session, session.channels, features, 0.02) for session in sessions]
        x = np.concatenate([day.x[np.isfinite(day.y).all(axis=1)] for day in days])
        y = np.concatenate([day.y[np.isfinite(day.y).all(axis=1)] for day in days])
        # When each day's blocks contribute sufficient statistics.
        moments = source_statistics(
            block
            for session in sessions
            for block in iter_design(session, session.channels, features, 0.02, block_size=19)
        )
        np.testing.assert_allclose(moments.yy, y.T @ y, rtol=1e-12, atol=1e-12)
        # Then the same standardized ridge decoder is recovered without source X.
        center, bias = moments.sum_x / moments.n, moments.sum_y / moments.n
        scale = np.sqrt(np.maximum(np.diag(moments.xx) / moments.n - center**2, 0))
        scale = np.where(scale > 1e-8, scale, 1)
        covariance = (moments.xx / moments.n - np.outer(center, center)) / np.outer(scale, scale)
        cross = (moments.xy / moments.n - np.outer(center, bias)) / scale[:, None]
        actual = np.linalg.solve(covariance + 2 * np.eye(len(center)), cross)
        expected = fit(x, y, 2)
        np.testing.assert_allclose(actual, expected.weights, atol=1e-12)
        np.testing.assert_allclose(center, expected.center, atol=1e-12)
        np.testing.assert_allclose(scale, expected.scale, atol=1e-12)
        self.assertEqual(moments.n, len(x))


if __name__ == "__main__":
    _ = unittest.main()
