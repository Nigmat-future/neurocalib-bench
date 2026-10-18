"""Cumulative research provenance, finite batches, and the pre-holdout seal."""

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from calib15.records import BASELINES, Gate
from calib15.records import Trial as ParentTrial
from calib15.schema import Calibration, Features, Row
from calibnext import workflow
from calibnext.records import require_final_access
from calibnext.schema import Batch, Config, Sanity
from neurocalib.ledger import Started, digest, environment, file_digest, write_record

ANCHOR = 0.35010946184569786


class CalibNextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parent = self.root / "studies/calib15"
        self.study = self.root / "studies/calibnext"
        self.results = self.study / "results"
        for package in ("calibnext", "calib15", "neurocalib"):
            (self.root / package).mkdir()
            (self.root / package / "core.py").write_text("CORE = 1\n", encoding="utf-8")
        self.results.mkdir(parents=True)
        (self.parent / "results").mkdir(parents=True)
        for directory in (self.study, self.parent):
            for name in ("PROTOCOL.md", "program.md"):
                (directory / name).write_text("Cumulative research.\n", encoding="utf-8")
        for name in ("pyproject.toml", "uv.lock"):
            (self.root / name).write_text("fixed\n", encoding="utf-8")
        files = [
            {"name": "indy_20160101_01.mat", "animal": "indy", "role": "source"},
            *(
                {"name": f"indy_201602{day:02d}_01.mat", "animal": "indy", "role": "development"}
                for day in range(1, 10)
            ),
            {"name": "indy_20160301_01.mat", "animal": "indy", "role": "test"},
        ]
        protocol = {
            "record_url": "https://zenodo.org/records/3854034",
            "files": [{**item, "size": 1, "md5": "0" * 32} for item in files],
        }
        (self.parent / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
        (self.study / "protocol.json").write_text('{"version":"calibnext-v1"}', encoding="utf-8")
        self.features = Features(lags=10, smoothing_seconds=0.06, source_alpha=10.0)
        write_record(self.parent / "results/features.json", self.features)
        (self.parent / "results/source_tuning.json").write_text("{}\n", encoding="utf-8")
        self.build_parent()

    def rows(self, score=ANCHOR):
        return tuple(
            Row(
                animal="indy",
                session=f"indy_201602{day:02d}_01.mat",
                budget_seconds=15,
                r2=score,
                r2_x=score,
                r2_y=score,
                rmse=1.0,
                frozen_r2=0.4 if day == 1 else 0.3,
                calibration_seconds=0.01,
                n_prefix_bins=750,
                n_scoring_bins=100,
                n_features=20,
                missing_channels=0,
                direction_bins=4,
                speed_std_mm_s=10.0,
            )
            for day in range(1, 10)
        )

    def build_parent(self):
        configs = [
            *BASELINES,
            *(Calibration(mode="subspace", strength=number / 9) for number in range(1, 9)),
        ]
        anchor = Calibration(mode="residual", alpha=10.0, strength=0.5)
        events = []
        for config in configs:
            start = ParentTrial(
                sequence=len(events) + 1,
                trial_id=digest(config),
                config=config,
                features=self.features,
                hypothesis="Previously exposed",
                kind="baseline" if config in BASELINES else "candidate",
                inputs={"calib15/core.py": file_digest(self.root / "calib15/core.py")},
                environment=environment(),
                started_at=datetime.now(UTC).isoformat(),
                status="running",
            )
            score = ANCHOR if config == anchor else 0.2
            events.extend(
                (
                    start,
                    start.model_copy(
                        update={
                            "sequence": start.sequence + 1,
                            "status": "keep",
                            "score": score,
                            "rows": self.rows(score),
                        }
                    ),
                )
            )
        ledger = self.parent / "results/experiments.jsonl"
        ledger.write_text(
            "".join(event.model_dump_json() + "\n" for event in events), encoding="utf-8"
        )
        write_record(
            self.parent / "results/decision.json",
            Gate(
                passed=False,
                reason="Eight earlier candidates failed",
                inputs={},
                ledger_sha256=file_digest(ledger),
                attempted_candidates=8,
                baseline_best=anchor,
                baseline_score=ANCHOR,
                baseline_negative_days=1,
                target_best=Calibration(mode="target", alpha=10.0),
            ),
        )
        write_record(
            self.results / "sanity.json",
            Sanity(matches_original=True, parent_trial_id=digest(anchor), rows=self.rows()),
        )

    def batch(self, name, configs, parent=""):
        batch = Batch(
            batch_id=name,
            configs=tuple(configs),
            rationale="Finite follow-up",
            parent_ledger_sha256=parent,
        )
        write_record(self.study / "batches" / f"{name}.json", batch, exclusive=True)
        return batch

    def evaluate(self, root, config, phase="development", budgets=(15,)):
        self.assertEqual((phase, budgets), ("development", (15,)))
        return self.rows(0.38 if config.alpha == 1 else 0.34)

    def test_first_passing_candidate_stops_batch(self):
        # Given: an immutable two-candidate follow-up batch.
        first, second = Config(alpha=1.0), Config(alpha=2.0)
        self.batch("001-map", (first, second))
        with patch("calibnext.evaluation.evaluate", side_effect=self.evaluate):
            workflow.run_trial(self.root, first, "Prior correction", "001-map")
            # When: a later candidate attempts to continue after success.
            with self.assertRaisesRegex(RuntimeError, "passed"):
                workflow.run_trial(self.root, second, "Unnecessary continuation", "001-map")
        # Then: the decision retains prior exposure rather than claiming independence.
        decision = workflow.gate(self.root)
        self.assertTrue(decision.passed)
        self.assertEqual(
            (decision.inherited_configurations, decision.inherited_candidates), (32, 8)
        )

    def test_failed_candidate_is_reused_and_next_batch_extends_history(self):
        # Given: a complete failed batch, with no unseen test access.
        first = Config(alpha=2.0)
        self.batch("001-map", (first,))
        with patch("calibnext.evaluation.evaluate", side_effect=self.evaluate) as evaluator:
            original = workflow.run_trial(self.root, first, "First hypothesis", "001-map")
            repeated = workflow.run_trial(self.root, first, "Repeated", "001-map")
            ledger = self.results / "experiments.jsonl"
            second = Config(alpha=3.0)
            self.batch("002-map", (second,), file_digest(ledger))
            # When: a new finite batch is appended to that exact ledger state.
            workflow.run_trial(self.root, second, "Additional hypothesis", "002-map")
        # Then: duplicate work is avoided and both exposures remain counted.
        self.assertEqual(original, repeated)
        self.assertEqual(evaluator.call_count, 2)
        self.assertEqual(workflow.gate(self.root).additional_attempts, 2)

    def test_failed_gate_cannot_be_frozen(self):
        # Given: a candidate below the inherited reference margin.
        config = Config(alpha=2.0)
        self.batch("001-map", (config,))
        with patch("calibnext.evaluation.evaluate", side_effect=self.evaluate):
            workflow.run_trial(self.root, config, "Small correction", "001-map")
        # When: selection tries to open a final-test seal.
        with self.assertRaisesRegex(RuntimeError, "gate"):
            workflow.freeze(self.root)
        # Then: no final access artifact exists.
        self.assertFalse((self.results / "frozen.json").exists())

    def test_manifest_cannot_change_after_its_first_exposure(self):
        # Given: one completed failure from an ordered two-candidate manifest.
        first, second = Config(alpha=2.0), Config(alpha=3.0)
        batch = self.batch("001-map", (first, second))
        with patch("calibnext.evaluation.evaluate", side_effect=self.evaluate):
            workflow.run_trial(self.root, first, "First hypothesis", "001-map")
        write_record(
            self.study / "batches/001-map.json", batch.model_copy(update={"rationale": "Edited"})
        )
        # When: a later hypothesis tries to use that edited declaration.
        with self.assertRaisesRegex(RuntimeError, "changed"):
            workflow.run_trial(self.root, second, "Second hypothesis", "001-map")
        # Then: original exposure cannot be silently redefined.

    def test_crash_is_preserved_and_cannot_restart(self):
        # Given: a declared candidate that fails during evaluation.
        config = Config(alpha=2.0)
        self.batch("001-map", (config,))
        with (
            patch("calibnext.evaluation.evaluate", side_effect=RuntimeError("failed")),
            self.assertRaisesRegex(RuntimeError, "failed"),
        ):
            workflow.run_trial(self.root, config, "Failure hypothesis", "001-map")
        # When: the same candidate is submitted again.
        with self.assertRaisesRegex(RuntimeError, "cannot restart"):
            workflow.run_trial(self.root, config, "Retry", "001-map")
        # Then: its crash remains a counted, terminal exposure.
        events = (self.results / "experiments.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(events[-1])["status"], "crash")

    def test_final_guard_checks_marker_and_complete_ledger_winner(self):
        # Given: a correctly frozen first passing candidate.
        config = Config(alpha=1.0)
        self.batch("001-map", (config,))
        with patch("calibnext.evaluation.evaluate", side_effect=self.evaluate):
            workflow.run_trial(self.root, config, "Passing hypothesis", "001-map")
        path = workflow.freeze(self.root)
        with self.assertRaisesRegex(RuntimeError, "marker"):
            require_final_access(self.root)
        write_record(self.results / "final.started.json", Started(frozen_sha256=file_digest(path)))
        require_final_access(self.root)
        frozen = json.loads(path.read_text(encoding="utf-8"))
        frozen["winner"]["hypothesis"] = "Not the recorded complete winner"
        path.write_text(json.dumps(frozen), encoding="utf-8")
        # When: a changed winner attempts to cross the final boundary.
        with self.assertRaisesRegex(RuntimeError, "changed"):
            require_final_access(self.root)
        # Then: plausible scores cannot replace the full ledger winner.


if __name__ == "__main__":
    unittest.main()
