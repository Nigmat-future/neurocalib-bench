"""The v2 evaluator keeps time budgets, animals and holdouts separate."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from calib15 import evaluation
from calib15.features import build
from calib15.records import Comparison, Frozen, Gate
from calib15.schema import Calibration, Features, Recording, Study
from neurocalib.schema import Session


class FeatureChecks(unittest.TestCase):
    def test_ema_uses_no_future_and_restarts_after_a_gap(self):
        # Given two neural segments, with a future jump and an intervening gap.
        session = Session(
            "example",
            np.array([0.02, 0.04, 0.06, 0.20, 0.22]),
            np.array([[0.0], [2.0], [100.0], [10.0], [0.0]]),
            np.ones((5, 2)),
            ("M1 001",),
        )
        # When a causal EMA is constructed independently for each segment.
        result = build(session, session.channels, Features(lags=1, smoothing_seconds=0.06), 0.02)
        # Then future activity cannot enter earlier features and the gap resets state.
        alpha = 1 - np.exp(-0.02 / 0.06)
        np.testing.assert_allclose(result.x[:2, 0], [0, 2 * alpha])
        np.testing.assert_allclose(result.x[3:, 0], [10, 10 * (1 - alpha)])

    def test_no_smoothing_preserves_the_current_and_past_bins(self):
        # Given a complete recording with three increasing counts.
        session = Session(
            "example",
            np.array([0.02, 0.04, 0.06]),
            np.array([[1.0], [2.0], [3.0]]),
            np.ones((3, 2)),
            ("M1 001",),
        )
        # When smoothing is disabled and two lags are requested.
        result = build(session, session.channels, Features(lags=2, smoothing_seconds=0.0), 0.02)
        # Then features contain the current count followed by the past count.
        np.testing.assert_array_equal(result.x, [[2, 1], [3, 2]])


class EvaluationChecks(unittest.TestCase):
    def setUp(self):
        self.evaluation = evaluation
        self.root = Path("unused-fixture-root")
        descriptions = [
            ("indy_20160930_05.mat", "indy", "source"),
            ("indy_20161005_06.mat", "indy", "source"),
            ("indy_20161011_03.mat", "indy", "development"),
            ("indy_20161220_02.mat", "indy", "test"),
            ("loco_20170210_03.mat", "loco", "source"),
            ("loco_20170215_02.mat", "loco", "test"),
        ]
        self.study = Study(
            record_url="https://zenodo.org/records/3854034",
            files=tuple(
                Recording(name=name, animal=animal, role=role, size=1, md5="0" * 32)
                for name, animal, role in descriptions
            ),
        )
        times = np.arange(1, 1801, dtype=np.float64) * 0.02
        counts = (2 + np.sin(times * 2))[:, None]
        self.sessions = {
            spec.name: Session(
                spec.name.removesuffix(".mat"),
                times,
                counts,
                np.column_stack((2 * counts[:, 0], -counts[:, 0]))
                * (1 if spec.animal == "indy" else -5),
                ("M1 001",),
            )
            for spec in self.study.files
        }
        self.addCleanup(patch.stopall)
        patch("calib15.evaluation.read_study", return_value=self.study).start()
        patch(
            "calib15.evaluation.recording_path", side_effect=lambda _, spec: Path(spec.name)
        ).start()
        self.loader = patch(
            "calib15.evaluation.load_session", side_effect=lambda path, _: self.sessions[path.name]
        ).start()
        self.features = Features(lags=1, source_alpha=1e-6)
        self.frozen = Frozen(
            decision=Gate(
                passed=True, reason="fixture", inputs={}, ledger_sha256="", attempted_candidates=0
            ),
            features=self.features,
            inputs={},
            ledger_sha256="",
            environment={},
            comparisons=(Comparison(label="frozen", config=Calibration(mode="frozen")),),
        )

    def test_prefix_labels_and_fixed_scoring_mask_are_isolated(self):
        # Given future labels unlike the prefix, plus missing prefix/scoring targets.
        name = "indy_20161011_03.mat"
        original = self.sessions[name]
        velocity = original.velocity.copy()
        velocity[original.times > 15] *= 100
        velocity[0] = np.nan
        velocity[np.flatnonzero(original.times >= 30.5)[0]] = np.nan
        self.sessions[name] = Session(
            name, original.times, original.counts, velocity, original.channels
        )
        with patch("calib15.evaluation.adapt", wraps=self.evaluation.adapt) as adapter:
            # When evaluating a supervised fifteen-second calibration.
            rows = self.evaluation.evaluate(
                self.root,
                self.features,
                Calibration(mode="residual"),
                "development",
            )
        # Then fitting sees only finite prefix labels and scoring uses the common suffix.
        prefix_y = adapter.call_args.args[2]
        self.assertEqual(len(prefix_y), 749)
        self.assertLess(float(np.max(np.abs(prefix_y))), 7)
        self.assertEqual(rows[0].n_prefix_bins, 749)
        self.assertEqual(rows[0].n_scoring_bins, int((original.times >= 30.5).sum()) - 1)

    def test_animals_use_separate_source_models(self):
        # Given opposite input/output mappings for identical channel names across animals.
        with (
            patch("calib15.records.require_final_access"),
            patch("calib15.records.read_frozen", return_value=self.frozen),
        ):
            # When the authorized test split is evaluated without adaptation.
            rows = self.evaluation.evaluate(
                self.root,
                self.features,
                Calibration(mode="frozen"),
                "test",
                budgets=(0, 5, 15, 30),
            )
        # Then each animal recovers its own source mapping.
        self.assertEqual({row.animal for row in rows}, {"indy", "loco"})
        self.assertTrue(all(row.r2 > 0.999 for row in rows))

    def test_test_gate_precedes_all_recording_access(self):
        # Given a final-access guard rejecting an unfrozen experiment.
        with (
            patch("calib15.records.require_final_access", side_effect=RuntimeError("locked")),
            self.assertRaisesRegex(RuntimeError, "locked"),
        ):
            # When test evaluation is requested.
            self.evaluation.evaluate(
                self.root,
                self.features,
                Calibration(mode="frozen"),
                "test",
            )
        # Then neither source nor held-out recordings were opened.
        self.loader.assert_not_called()

    def test_started_test_marker_does_not_allow_new_configuration_or_partial_budgets(self):
        # Given authorized test access bound to an explicit feature and comparison seal.
        variants = [
            (Features(lags=2), Calibration(mode="frozen"), (0, 5, 15, 30)),
            (self.features, Calibration(mode="residual"), (0, 5, 15, 30)),
            (self.features, Calibration(mode="frozen"), (15,)),
        ]
        with (
            patch("calib15.records.require_final_access"),
            patch("calib15.records.read_frozen", return_value=self.frozen),
        ):
            for features, calibration, budgets in variants:
                # When a caller changes an aspect of the sealed test comparison.
                with (
                    self.subTest(features=features, calibration=calibration, budgets=budgets),
                    self.assertRaisesRegex(RuntimeError, "must match"),
                ):
                    self.evaluation.evaluate(self.root, features, calibration, "test", budgets)
        # Then all requests are refused before any source or target is loaded.
        self.loader.assert_not_called()

    def test_source_tuning_cannot_change_after_the_first_trial(self):
        # Given a development ledger that already fixes the shared source features.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "studies/calib15/results"
            output.mkdir(parents=True)
            (output / "experiments.jsonl").touch()
            # When source tuning is requested again.
            with self.assertRaisesRegex(ValueError, "first trial"):
                self.evaluation.tune_source(root)
        # Then not even the source recordings were opened.
        self.loader.assert_not_called()

    def test_source_tuning_reads_only_indy_sources_and_logs_every_combination(self):
        # Given development, test and external-animal files alongside two Indy sources.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # When the fixed source-only preprocessing grid is selected.
            chosen = self.evaluation.tune_source(root)
            evidence = json.loads((root / "studies/calib15/results/source_tuning.json").read_text())
            saved = Features.model_validate_json(
                (root / "studies/calib15/results/features.json").read_bytes()
            )
        # Then all thirty candidates have both folds, without opening any target file.
        self.assertEqual(chosen, saved)
        self.assertEqual(len(evidence["trials"]), 30)
        self.assertEqual(
            {trial["features"]["source_alpha"] for trial in evidence["trials"]},
            {0.1, 1.0, 10.0, 100.0, 1000.0},
        )
        self.assertTrue(all(len(trial["scores"]) == 2 for trial in evidence["trials"]))
        self.assertEqual(
            {call.args[0].name for call in self.loader.call_args_list},
            {"indy_20160930_05.mat", "indy_20161005_06.mat"},
        )


if __name__ == "__main__":
    unittest.main()
