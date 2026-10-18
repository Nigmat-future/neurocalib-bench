"""Research boundaries use tiny files and a deterministic evaluation substitute."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neurocalib import research
from neurocalib.ledger import environment
from neurocalib.reproduce import reproduce
from neurocalib.schema import Strategy


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "neurocalib").mkdir()
        (self.root / "neurocalib/core.py").write_text("CORE = 1\n", encoding="utf-8")
        for name in ("pyproject.toml", "uv.lock", "PROTOCOL.md"):
            (self.root / name).write_text("fixed\n", encoding="utf-8")
        protocol = {
            "version": "test-v1",
            "record_url": "https://example.org/record",
            "bin_seconds": 0.02,
            "score_start_seconds": 30.5,
            "budgets_seconds": [0, 5, 15, 30],
            "files": [
                {"name": f"indy_2016100{number}_01.mat", "role": role, "size": 1, "md5": "0" * 32}
                for number, role in enumerate(
                    ("source", "development", "development", "test", "test"), start=1
                )
            ],
        }
        (self.root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")

    def candidate(self, name="candidate", **changes):
        path = self.root / f"{name}.json"
        path.write_text(Strategy(**changes).model_dump_json(), encoding="utf-8")
        return path

    def evaluate(self, root, strategy, phase):
        protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
        return [
            {
                "session": item["name"],
                "mode": strategy.mode,
                "budget_seconds": budget,
                "r2": strategy.strength / 2,
                "rmse": 1.0,
                "inference_us": 2.0,
                "calibration_seconds": 0.01,
                "missing_channels": 0,
                "r2_x": strategy.strength / 2,
                "r2_y": strategy.strength / 2,
                "zero_r2": 0.0,
                "zero_rmse": 2.0,
                "n_scoring_bins": 200,
                "n_prefix_bins": budget * 50,
                "n_channels": 2,
                "n_features": 20,
                "last_bin_seconds": 35.0,
            }
            for item in protocol["files"]
            if item["role"] == phase
            for budget in protocol["budgets_seconds"]
        ]

    def ledger(self):
        return [
            json.loads(line)
            for line in (self.root / "results/experiments.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    def test_duplicate_reuses_result_without_spending_trial(self):
        # Given: one completed development experiment.
        candidate = self.candidate()
        with patch("neurocalib.research.evaluate", side_effect=self.evaluate) as evaluator:
            original = research.record_trial(self.root, candidate, "Baseline")
            # When: the same strategy is requested with different prose.
            duplicate = research.record_trial(self.root, candidate, "Repeated hypothesis")
        # Then: evaluation runs once and the original result is returned.
        self.assertEqual(evaluator.call_count, 1)
        self.assertEqual(original, duplicate)

    def test_crash_remains_in_ledger_and_propagates(self):
        # Given: an experiment whose evaluator fails.
        candidate = self.candidate()
        # When: the candidate runs.
        with (
            patch("neurocalib.research.evaluate", side_effect=RuntimeError("broken run")),
            self.assertRaisesRegex(RuntimeError, "broken run"),
        ):
            research.record_trial(self.root, candidate, "Failure hypothesis")
        # Then: the append-only ledger preserves both the start and failure.
        self.assertEqual([entry["status"] for entry in self.ledger()], ["running", "crash"])

    def test_budget_cannot_be_increased_after_search_starts(self):
        # Given: the sole allowed trial has completed.
        with patch("neurocalib.research.evaluate", side_effect=self.evaluate):
            research.record_trial(self.root, self.candidate(), "Baseline", max_trials=1)
            # When: a second configuration attempts to increase the budget.
            with self.assertRaisesRegex(research.ResearchError, "budget"):
                research.record_trial(
                    self.root, self.candidate("second", alpha=2.0), "More", max_trials=2
                )
        # Then: no extra candidate was recorded.
        self.assertEqual(len(self.ledger()), 2)

    def test_immutable_code_change_blocks_selection(self):
        # Given: a completed experiment followed by a core modification.
        with patch("neurocalib.research.evaluate", side_effect=self.evaluate):
            research.record_trial(self.root, self.candidate(), "Baseline")
        (self.root / "neurocalib/core.py").write_text("CORE = 2\n", encoding="utf-8")
        # When: attempting to select the winner.
        with self.assertRaisesRegex(research.ResearchError, "changed"):
            research.freeze(self.root)
        # Then: no final lock exists.
        self.assertFalse((self.root / "results/frozen.json").exists())

    def test_final_failure_resumes_same_frozen_candidates(self):
        # Given: a frozen winner and a failure after the first final result.
        with patch("neurocalib.research.evaluate", side_effect=self.evaluate):
            research.record_trial(self.root, self.candidate(mode="center"), "Centering")
        research.freeze(self.root)
        calls = 0

        def fail_second(root, strategy, phase):
            nonlocal calls
            calls += 1
            if calls == 2:
                message = "interrupted final"
                raise RuntimeError(message)
            return self.evaluate(root, strategy, phase)

        with (
            patch("neurocalib.research.evaluate", side_effect=fail_second),
            self.assertRaisesRegex(RuntimeError, "interrupted final"),
        ):
            research.final_test(self.root)
        # When: final evaluation resumes and is requested again after completion.
        with patch("neurocalib.research.evaluate", side_effect=self.evaluate) as evaluator:
            output = research.final_test(self.root)
            repeated = research.final_test(self.root)
        # Then: completed strategies are not re-evaluated and the lock stays exclusive.
        self.assertEqual(output, repeated)
        self.assertEqual(evaluator.call_count, 6)
        with self.assertRaises((research.ResearchError, FileExistsError)):
            research.freeze(self.root)

    def test_supervised_reference_cannot_enter_search(self):
        # Given: a supervised candidate intended only as a final reference.
        candidate = self.candidate(mode="supervised")
        # When: attempting development search.
        with self.assertRaisesRegex(research.ResearchError, "supervised"):
            research.record_trial(self.root, candidate, "Invalid search")
        # Then: no ledger was created.
        self.assertFalse((self.root / "results/experiments.jsonl").exists())

    def test_zero_budget_improvement_cannot_change_winner(self):
        # Given: a frozen baseline and an alternative with lower positive-budget R2.
        def misleading(root, strategy, phase):
            rows = self.evaluate(root, strategy, phase)
            for row in rows:
                if row["budget_seconds"] == 0:
                    row["r2"] = 100000.0
            return rows

        with patch("neurocalib.research.evaluate", side_effect=misleading):
            research.record_trial(self.root, self.candidate(), "Frozen baseline")
            research.record_trial(
                self.root, self.candidate("second", mode="center", strength=0.4), "Centering"
            )
        # When: winner selection uses the predeclared objective.
        path = research.freeze(self.root)
        # Then: a flattering zero-budget result cannot win the search.
        frozen = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(frozen["winner"]["strategy"]["mode"], "frozen")
        self.assertEqual(self.ledger()[-1]["status"], "discard")

    def test_reproduction_allows_platform_change_without_reselecting(self):
        # Given: a completed final comparison and a different platform.
        with patch("neurocalib.research.evaluate", side_effect=self.evaluate):
            research.record_trial(self.root, self.candidate(), "Baseline")
            research.freeze(self.root)
            research.final_test(self.root)
        runtime = {**environment(), "platform": "another-platform"}

        def different_timing(root, strategy, phase):
            rows = self.evaluate(root, strategy, phase)
            for row in rows:
                row["inference_us"] = 100000.0
                row["calibration_seconds"] = 100.0
            return rows

        # When: the exact frozen comparison is reproduced.
        with (
            patch("neurocalib.research.evaluate", side_effect=different_timing) as evaluator,
            patch("neurocalib.reproduce.environment", return_value=runtime),
        ):
            output = reproduce(self.root)
        # Then: all seven declared comparisons rerun, recording the new environment.
        reproduced = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(reproduced["matches"])
        self.assertEqual(evaluator.call_count, 7)
        self.assertEqual(reproduced["environment"]["platform"], "another-platform")

    def test_reproduction_preserves_evidence_when_scientific_results_differ(self):
        # Given: a completed final comparison and an inconsistent rerun.
        with patch("neurocalib.research.evaluate", side_effect=self.evaluate):
            research.record_trial(self.root, self.candidate(), "Baseline")
            research.freeze(self.root)
            original_path = research.final_test(self.root)
        original = original_path.read_bytes()

        def changed_metric(root, strategy, phase):
            rows = self.evaluate(root, strategy, phase)
            rows[0]["r2"] = -10.0
            return rows

        # When: a scientific metric changes beyond tolerance.
        with (
            patch("neurocalib.research.evaluate", side_effect=changed_metric),
            self.assertRaisesRegex(research.ResearchError, "reproduction differs"),
        ):
            reproduce(self.root)
        # Then: mismatch evidence is saved and the original final report is intact.
        output = self.root / "results/reproduced/report.json"
        self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["matches"])
        self.assertEqual(original_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
