"""Small deterministic checks of the calibration study's research boundaries."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calib15 import research
from calib15.records import BASELINES
from calib15.schema import Calibration, Features, Row


class Calib15ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = self.root / "studies/calib15"
        self.results = self.study / "results"
        self.results.mkdir(parents=True)
        for package in ("calib15", "neurocalib"):
            (self.root / package).mkdir()
            (self.root / package / "core.py").write_text("CORE = 1\n", encoding="utf-8")
        for name in ("pyproject.toml", "uv.lock"):
            (self.root / name).write_text("fixed\n", encoding="utf-8")
        for name in ("PROTOCOL.md", "program.md"):
            (self.study / name).write_text("Fixed study contract.\n", encoding="utf-8")
        files = [
            {"name": "indy_20160101_01.mat", "animal": "indy", "role": "source"},
            {"name": "loco_20160101_01.mat", "animal": "loco", "role": "source"},
            *(
                {"name": f"indy_201602{day:02d}_01.mat", "animal": "indy", "role": "development"}
                for day in range(1, 10)
            ),
            {"name": "indy_20160301_01.mat", "animal": "indy", "role": "test"},
            {"name": "loco_20160301_01.mat", "animal": "loco", "role": "test"},
        ]
        self.protocol = {
            "record_url": "https://zenodo.org/records/3854034",
            "files": [{**item, "size": 1, "md5": "0" * 32} for item in files],
        }
        (self.study / "protocol.json").write_text(json.dumps(self.protocol), encoding="utf-8")
        (self.results / "features.json").write_text(Features().model_dump_json(), encoding="utf-8")
        (self.results / "source_tuning.json").write_text("{}\n", encoding="utf-8")

    def evaluate(self, root, features, config, phase, budgets=(15,)):
        protocol = json.loads((root / "studies/calib15/protocol.json").read_text(encoding="utf-8"))
        value = 0.5 + (0.03 * config.strength if config.mode in ("affine", "subspace") else 0)
        return tuple(
            Row(
                animal=item["animal"],
                session=item["name"],
                budget_seconds=budget,
                r2=value,
                r2_x=value,
                r2_y=value,
                rmse=1.0,
                frozen_r2=0.4,
                calibration_seconds=0.01,
                n_prefix_bins=budget * 50,
                n_scoring_bins=100,
                n_features=20,
                missing_channels=0,
                direction_bins=4,
                speed_std_mm_s=10.0,
            )
            for item in protocol["files"]
            if item["role"] == phase
            for budget in budgets
        )

    def baselines(self):
        for config in BASELINES:
            research.run_trial(self.root, config, "Predeclared baseline", "baseline")

    def test_failed_gate_blocks_freeze_and_test_access(self):
        # Given: all strong baselines and a candidate below the declared margin.
        with patch("calib15.research.evaluate", side_effect=self.evaluate):
            self.baselines()
            research.run_trial(
                self.root, Calibration(mode="affine", strength=0.1), "Small correction", "candidate"
            )
        # When: the development decision is made.
        decision = research.gate(self.root)
        # Then: failed development cannot access held-out data.
        self.assertFalse(decision.passed)
        with self.assertRaises(research.ResearchError):
            research.freeze(self.root)
        with self.assertRaises((research.ResearchError, FileNotFoundError)):
            research.require_final_access(self.root)
        self.assertTrue((self.results / "decision.json").exists())

    def test_duplicate_does_not_consume_another_candidate(self):
        # Given: a successful candidate with a completed baseline sweep.
        config = Calibration(mode="affine")
        with patch("calib15.research.evaluate", side_effect=self.evaluate) as evaluator:
            self.baselines()
            original = research.run_trial(self.root, config, "Affine correction", "candidate")
            # When: identical content is submitted under different prose.
            repeated = research.run_trial(self.root, config, "Repeated idea", "candidate")
        # Then: only the original attempt is counted and evaluated.
        self.assertEqual(original, repeated)
        self.assertEqual(evaluator.call_count, len(BASELINES) + 1)
        self.assertEqual(research.gate(self.root).attempted_candidates, 1)

    def test_code_tamper_blocks_trial_reuse(self):
        # Given: completed baselines followed by a core-file modification.
        with patch("calib15.research.evaluate", side_effect=self.evaluate):
            self.baselines()
        (self.root / "calib15/core.py").write_text("CORE = 2\n", encoding="utf-8")
        # When: another hypothesis is attempted.
        with self.assertRaisesRegex(research.ResearchError, "changed"):
            research.run_trial(self.root, Calibration(mode="affine"), "New idea", "candidate")
        # Then: no candidate was evaluated.

    def test_final_marker_precedes_test_evaluation(self):
        # Given: a candidate passing the development gate.
        with patch("calib15.research.evaluate", side_effect=self.evaluate):
            self.baselines()
            research.run_trial(
                self.root, Calibration(mode="affine"), "Affine correction", "candidate"
            )
        research.freeze(self.root)
        observed = []

        def guarded(root, features, config, phase, budgets=(15,)):
            research.require_final_access(root)
            observed.append(phase)
            return self.evaluate(root, features, config, phase, budgets)

        # When: the frozen final comparison runs.
        with patch("calib15.seal.evaluate", side_effect=guarded):
            output = research.final(self.root)
        # Then: all four test methods were guarded and the frozen result is complete.
        self.assertEqual(observed, ["test"] * 4)
        self.assertTrue(output.exists())

    def passing_freeze(self):
        with patch("calib15.research.evaluate", side_effect=self.evaluate):
            self.baselines()
            research.run_trial(self.root, Calibration(mode="affine"), "Correction", "candidate")
        return research.freeze(self.root)

    def test_candidate_limit_excludes_baseline_sweep(self):
        # Given: all strong baseline configurations and twelve distinct candidate attempts.
        with patch("calib15.research.evaluate", side_effect=self.evaluate) as evaluator:
            self.baselines()
            for number in range(1, 13):
                research.run_trial(
                    self.root,
                    Calibration(mode="affine", strength=number / 13),
                    "Bounded correction",
                    "candidate",
                )
            # When: a thirteenth candidate is proposed.
            with self.assertRaisesRegex(research.ResearchError, "budget"):
                research.run_trial(self.root, Calibration(mode="subspace"), "Extra", "candidate")
        # Then: baseline comparisons did not spend the twelve-candidate allowance.
        self.assertEqual(evaluator.call_count, len(BASELINES) + 12)

    def test_crash_is_logged_and_cannot_be_restarted(self):
        # Given: completed strong baselines and a candidate that crashes.
        with patch("calib15.research.evaluate", side_effect=self.evaluate):
            self.baselines()
        config = Calibration(mode="affine")
        with (
            patch("calib15.research.evaluate", side_effect=RuntimeError("crashed")),
            self.assertRaisesRegex(RuntimeError, "crashed"),
        ):
            research.run_trial(self.root, config, "Failure evidence", "candidate")
        # When: that exact failed candidate is submitted again.
        with self.assertRaisesRegex(research.ResearchError, "cannot restart"):
            research.run_trial(self.root, config, "Retry", "candidate")
        # Then: the terminal failure remains in the append-only ledger.
        lines = (self.results / "experiments.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(lines[-1])["status"], "crash")

    def test_extra_negative_transfer_day_fails_development_gate(self):
        # Given: a candidate with large average gain but harm on one day.
        def harmful(root, features, config, phase, budgets=(15,)):
            rows = self.evaluate(root, features, config, phase, budgets)
            if config.mode == "affine":
                return tuple(
                    row.model_copy(update={"r2": 0.39 if index == 0 else 0.56})
                    for index, row in enumerate(rows)
                )
            return rows

        with patch("calib15.research.evaluate", side_effect=harmful):
            self.baselines()
            research.run_trial(self.root, Calibration(mode="affine"), "Risky gain", "candidate")
        # When: the development gate evaluates its complete conditions.
        decision = research.gate(self.root)
        # Then: mean gain cannot hide an increased negative-transfer count.
        self.assertGreater(decision.delta, 0.02)
        self.assertFalse(decision.passed)

    def test_one_animal_loss_fails_final_despite_macro_gain(self):
        # Given: a passing development seal, then opposite outcomes across animals.
        self.passing_freeze()

        def mixed(root, features, config, phase, budgets=(15,)):
            rows = self.evaluate(root, features, config, phase, budgets)
            if config.mode == "affine":
                return tuple(
                    row.model_copy(update={"r2": 0.6 if row.animal == "indy" else 0.49})
                    for row in rows
                )
            return rows

        # When: all four frozen methods complete their final evaluation.
        with patch("calib15.seal.evaluate", side_effect=mixed):
            output = research.final(self.root)
        # Then: animal-macro gain cannot conceal the external animal's loss.
        result = json.loads(output.read_text(encoding="utf-8"))["outcome"]
        self.assertGreater(result["macro_delta"], 0.02)
        self.assertFalse(result["passed"])

    def test_semantically_duplicate_candidates_are_rejected(self):
        # Given: zero-strength corrections and unused affine component changes.
        configs = (
            Calibration(mode="affine", strength=0.0),
            Calibration(mode="subspace", strength=0.0),
            Calibration(mode="affine", components=7),
        )
        # When: those configurations attempt to enter the candidate budget.
        for config in configs:
            with self.subTest(config=config), self.assertRaises(research.ResearchError):
                research.run_trial(self.root, config, "Disguised duplicate", "candidate")
        # Then: no attempt was created for a frozen equivalent or unused parameter.
        self.assertFalse((self.results / "experiments.jsonl").exists())

    def test_failed_gate_development_can_be_reproduced_without_test_access(self):
        # Given: a failed development gate with no final seal or start marker.
        with patch("calib15.research.evaluate", side_effect=self.evaluate):
            self.baselines()
            research.run_trial(
                self.root, Calibration(mode="affine", strength=0.1), "Small correction", "candidate"
            )
        self.assertFalse(research.gate(self.root).passed)
        # When: completed development configurations are reproduced.
        with patch("calib15.replay.evaluate", side_effect=self.evaluate) as evaluator:
            output = research.reproduce(self.root)
        # Then: every completed development experiment reruns while test access stays sealed.
        self.assertEqual(evaluator.call_count, len(BASELINES) + 1)
        self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["matches_original"])
        self.assertFalse((self.results / "final.started.json").exists())

    def test_incomplete_cached_final_result_is_rejected(self):
        # Given: a completed final run with one accidentally removed scoring cell.
        self.passing_freeze()
        with patch("calib15.seal.evaluate", side_effect=self.evaluate):
            research.final(self.root)
        path = self.results / "final/winner.json"
        cached = json.loads(path.read_text(encoding="utf-8"))
        cached["rows"].pop()
        path.write_text(json.dumps(cached), encoding="utf-8")
        # When: final evaluation tries to reuse its cache.
        with self.assertRaisesRegex(research.ResearchError, "Evaluation cells"):
            research.final(self.root)
        # Then: a missing day/budget cannot be accepted as a complete final result.

    def test_original_eight_baselines_no_longer_authorize_candidates(self):
        # Given: only the initial regularization range, with no residual shrinkage sweep.
        with patch("calib15.research.evaluate", side_effect=self.evaluate):
            for mode in ("target", "residual"):
                for alpha in (0.01, 0.1, 1.0, 10.0):
                    research.run_trial(
                        self.root,
                        Calibration(mode=mode, alpha=alpha),
                        "Initial range only",
                        "baseline",
                    )
            # When: a candidate attempts to bypass the expanded strong comparison.
            with self.assertRaisesRegex(research.ResearchError, "every strong baseline"):
                research.run_trial(self.root, Calibration(mode="affine"), "Too early", "candidate")
        # Then: all 24 declared comparisons, including ridge and shrinkage extremes, are needed.
        self.assertEqual(len(BASELINES), 24)
        self.assertIn(Calibration(mode="target", alpha=1000.0), BASELINES)
        self.assertIn(Calibration(mode="residual", alpha=1000.0, strength=0.25), BASELINES)


if __name__ == "__main__":
    unittest.main()
