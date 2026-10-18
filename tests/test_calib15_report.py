"""Saved-record reporting works without raw recordings or a scientific evaluator."""

import csv
import tempfile
import unittest
from pathlib import Path
from statistics import fmean

from calib15.records import Outcome, Report, Run, Trial, append_event
from calib15.report import render
from calib15.schema import Calibration, Features, Row
from neurocalib.ledger import digest


class ReportChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results = self.root / "studies/calib15/results"
        self.results.mkdir(parents=True)
        self.ledger = self.results / "experiments.jsonl"
        self.sequence = 0

    def trial(self, config, kind, scores=(), *, status="discard"):
        rows = tuple(
            Row(
                animal="indy",
                session=f"indy_201610{day:02d}_01.mat",
                budget_seconds=15,
                r2=score,
                r2_x=score + 0.01,
                r2_y=score - 0.01,
                rmse=1.5,
                frozen_r2=0.45,
                calibration_seconds=0.001,
                n_prefix_bins=750,
                n_scoring_bins=1000,
                n_features=20,
                missing_channels=0,
                direction_bins=4,
                speed_std_mm_s=8.0,
            )
            for day, score in enumerate(scores, start=11)
        )
        start = Trial(
            sequence=self.sequence + 1,
            trial_id=digest(config),
            config=config,
            features=Features(),
            hypothesis="Synthetic reporting record",
            kind=kind,
            inputs={},
            environment={},
            started_at="2026-10-18T12:00:00Z",
            status="running",
        )
        append_event(self.ledger, start)
        terminal = start.model_copy(
            update={
                "sequence": self.sequence + 2,
                "status": status,
                "rows": rows,
                "score": fmean(scores) if scores else None,
                "error": "Synthetic interrupted trial" if status == "crash" else None,
            }
        )
        append_event(self.ledger, terminal)
        self.sequence += 2
        return terminal

    def test_all_terminal_configs_and_day_rows_are_exported_without_raw_access(self):
        # Given completed baselines, one candidate and one failed attempt, without raw files.
        self.trial(Calibration(mode="target", alpha=0.1), "baseline", (0.3, 0.4))
        self.trial(Calibration(mode="target"), "baseline", (0.4, 0.42))
        self.trial(Calibration(mode="residual", strength=0.5), "baseline", (0.46, 0.48))
        candidate = self.trial(Calibration(mode="affine"), "candidate", (0.5, 0.52))
        self.trial(Calibration(mode="subspace"), "candidate", status="crash")
        before = self.ledger.read_bytes()
        # When rendering only the retained evidence.
        image = render(self.root)
        with (self.results / "search_summary.csv").open(newline="", encoding="utf-8") as stream:
            summary = list(csv.DictReader(stream))
        with (self.results / "development.csv").open(newline="", encoding="utf-8") as stream:
            days = list(csv.DictReader(stream))
        # Then all terminal attempts and complete scientific rows survive unchanged.
        self.assertEqual(len(summary), 5)
        self.assertEqual([int(row["sequence"]) for row in summary], [2, 4, 6, 8, 10])
        self.assertEqual(summary[0]["negative_days"], "2")
        self.assertEqual(summary[-1]["status"], "crash")
        self.assertEqual(summary[-1]["score"], "")
        self.assertEqual(len(days), 8)
        self.assertEqual(days[-1]["trial_id"], candidate.trial_id)
        self.assertEqual(days[-1]["mode"], "affine")
        self.assertEqual(float(days[-1]["r2_x"]), 0.53)
        self.assertEqual(image, self.results / "calibration_comparison.png")
        self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(self.ledger.read_bytes(), before)
        self.assertFalse((self.root / "data").exists())
        self.assertFalse((self.results / "decision.json").exists())

    def test_empty_ledger_and_pending_candidate_still_produce_a_report(self):
        # Given no trials, followed by only one completed baseline.
        empty = render(self.root)
        self.assertTrue(empty.is_file())
        self.trial(Calibration(mode="target"), "baseline", (0.4, 0.42))
        # When candidate comparison must remain pending.
        pending = render(self.root)
        # Then reporting works without inventing a candidate or opening a final gate.
        self.assertTrue(pending.is_file())
        self.assertFalse((self.results / "final.started.json").exists())

    def test_completed_final_json_also_produces_per_animal_budget_curves(self):
        # Given already recorded results for both animals and four locked comparisons.
        trial = self.trial(Calibration(mode="affine"), "candidate", (0.5, 0.52))
        rows = tuple(
            trial.rows[0].model_copy(update={"animal": animal, "budget_seconds": budget})
            for animal in ("indy", "loco")
            for budget in (0, 5, 15, 30)
        )
        report = Report(
            frozen_sha256="saved-evidence",
            environment={},
            runs=tuple(
                Run(label=label, config=Calibration(mode=mode), frozen_sha256="saved", rows=rows)
                for label, mode in (
                    ("winner", "affine"),
                    ("target_best", "target"),
                    ("residual_best", "residual"),
                    ("frozen", "frozen"),
                )
            ),
            outcome=Outcome(passed=False, macro_delta=0, animals=()),
        )
        final = self.results / "final"
        final.mkdir()
        original = report.model_dump_json()
        (final / "report.json").write_text(original, encoding="utf-8")
        # When plotting existing final evidence.
        render(self.root)
        # Then both summaries exist and scientific evidence is never rewritten.
        self.assertTrue((final / "calibration_curves.png").is_file())
        self.assertEqual((final / "report.json").read_text(encoding="utf-8"), original)
        self.assertFalse((self.root / "data").exists())


if __name__ == "__main__":
    unittest.main()
