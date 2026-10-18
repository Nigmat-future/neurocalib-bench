"""Development-only reproduction retains mismatches and rejects changing evidence."""

import unittest
from pathlib import Path
from typing import override
from unittest.mock import patch

from calib15.schema import Row
from calibnext import cli, replay, workflow
from calibnext.records import append_event, read_events
from calibnext.schema import Batch, Config, Trial
from neurocalib.ledger import ResearchError, digest, write_record
from tests import test_calibnext_workflow as fixture


class ReplayChecks(unittest.TestCase):
    def __init__(self, method_name: str = "runTest") -> None:
        super().__init__(method_name)
        self.fixture: fixture.CalibNextTests = fixture.CalibNextTests()

    @override
    def setUp(self) -> None:
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    @property
    def root(self) -> Path:
        return self.fixture.root

    @property
    def results(self) -> Path:
        return self.fixture.results

    @property
    def output(self) -> Path:
        return self.results / "reproduced/development.json"

    def evaluate(
        self,
        root: Path,
        config: Config,
        phase: str = "development",
        budgets: tuple[int, ...] = (15,),
    ) -> tuple[Row, ...]:
        self.assertEqual((root, phase, budgets), (self.root, "development", (15,)))
        return self.fixture.rows(0.38 if config.alpha == 1 else 0.34)

    def completed(self, configs: tuple[Config, ...]) -> Trial:
        write_record(
            self.fixture.study / "batches/001-map.json",
            Batch(batch_id="001-map", configs=configs, rationale="Replay behavior check"),
        )
        with patch("calibnext.evaluation.evaluate", side_effect=self.evaluate):
            path = workflow.run_trial(self.root, configs[0], "Recorded development", "001-map")
        return Trial.model_validate_json(path.read_bytes())

    def pending(self, previous: Trial, config: Config, sequence: int) -> Trial:
        trial = previous.model_copy(
            update={
                "sequence": sequence,
                "trial_id": digest(config),
                "config": config,
                "status": "running",
                "score": None,
                "rows": (),
                "error": None,
            }
        )
        append_event(self.results / "experiments.jsonl", trial)
        return trial

    def test_replay_uses_only_completed_development_without_final_access(self) -> None:
        # Given one completed trial, one crash, and one pending declared configuration.
        configs = (Config(alpha=2), Config(alpha=3), Config(alpha=4))
        complete = self.completed(configs)
        with (
            patch("calibnext.evaluation.evaluate", side_effect=RuntimeError("synthetic crash")),
            self.assertRaisesRegex(RuntimeError, "synthetic crash"),
        ):
            _ = workflow.run_trial(self.root, configs[1], "Recorded crash", "001-map")
        _ = self.pending(complete, configs[2], 5)
        with (
            patch("calibnext.replay.evaluate", side_effect=self.evaluate) as evaluator,
            patch("calibnext.evaluation.load_session") as loader,
            patch("calibnext.records.require_final_access") as access,
        ):
            # When existing development evidence is reproduced.
            output = replay.reproduce_development(self.root)
        # Then only the completed configuration is rerun and no holdout boundary is crossed.
        result = replay.Reproduction.model_validate_json(output.read_bytes())
        evaluator.assert_called_once_with(self.root, configs[0])
        loader.assert_not_called()
        access.assert_not_called()
        self.assertTrue(result.matches_original)
        self.assertEqual(tuple(trial.config for trial in result.trials), (configs[0],))
        self.assertFalse((self.results / "frozen.json").exists())
        self.assertFalse((self.results / "final.started.json").exists())

    def test_scientific_mismatch_is_retained_before_raising(self) -> None:
        # Given an existing result and a replay with changed scientific scores.
        _ = self.completed((Config(alpha=2),))
        changed = self.fixture.rows(0.33)
        with (
            patch("calibnext.replay.evaluate", return_value=changed),
            self.assertRaisesRegex(ResearchError, "differs; evidence retained"),
        ):
            # When reproduction detects a numerical mismatch.
            _ = replay.reproduce_development(self.root)
        # Then failed evidence remains explicit and cannot be mistaken for success.
        result = replay.Reproduction.model_validate_json(self.output.read_bytes())
        self.assertFalse(result.matches_original)
        self.assertFalse(result.trials[0].matches_original)
        self.assertEqual(result.trials[0].rows, changed)

    def test_midrun_code_change_cannot_write_successful_reproduction(self) -> None:
        # Given a completed result and a bound scientific source file.
        _ = self.completed((Config(alpha=2),))

        def changed(root: Path, config: Config) -> tuple[Row, ...]:
            _ = (self.root / "calibnext/core.py").write_text("CORE = 2\n", encoding="utf-8")
            return self.evaluate(root, config)

        with (
            patch("calibnext.replay.evaluate", side_effect=changed),
            self.assertRaisesRegex(ResearchError, "changed during"),
        ):
            # When that code changes during numerical evaluation.
            _ = replay.reproduce_development(self.root)
        # Then no successful report is published.
        self.assertFalse(self.output.exists())

    def test_midrun_ledger_change_cannot_write_successful_reproduction(self) -> None:
        # Given a completed configuration and a second declaration not yet exposed.
        configs = (Config(alpha=2), Config(alpha=3))
        complete = self.completed(configs)

        def changed(root: Path, config: Config) -> tuple[Row, ...]:
            _ = self.pending(complete, configs[1], 3)
            return self.evaluate(root, config)

        with (
            patch("calibnext.replay.evaluate", side_effect=changed),
            self.assertRaisesRegex(ResearchError, "changed during"),
        ):
            # When another ledger exposure appears during reproduction.
            _ = replay.reproduce_development(self.root)
        # Then a partial snapshot cannot be published as complete reproduction.
        self.assertFalse(self.output.exists())

    def test_ledger_snapshot_is_bound_before_trial_selection(self) -> None:
        # Given a legal second completion appearing immediately after the first ledger read.
        configs = (Config(alpha=2), Config(alpha=3))
        complete = self.completed(configs)

        def changed(root: Path) -> tuple[Trial, ...]:
            events = read_events(root)
            start = self.pending(complete, configs[1], 3)
            append_event(
                self.results / "experiments.jsonl",
                start.model_copy(
                    update={
                        "sequence": 4,
                        "status": "discard",
                        "score": complete.score,
                        "rows": complete.rows,
                    }
                ),
            )
            return events

        with (
            patch("calibnext.replay.read_events", side_effect=changed),
            patch("calibnext.replay.evaluate", side_effect=self.evaluate),
            self.assertRaisesRegex(ResearchError, "changed"),
        ):
            # When reproduction could otherwise bind a newer hash to the older parsed snapshot.
            _ = replay.reproduce_development(self.root)
        # Then it cannot claim the newly completed ledger was fully reproduced.
        self.assertFalse(self.output.exists())

    def test_sanity_cannot_rewrite_bound_evidence_after_first_trial(self) -> None:
        # Given sanity evidence already bound by a completed trial.
        _ = self.completed((Config(alpha=2),))
        path = self.results / "sanity.json"
        original = path.read_bytes()
        with (
            patch("calibnext.cli.evaluate_reference") as evaluator,
            self.assertRaisesRegex(ResearchError, "immutable"),
        ):
            # When the CLI sanity action is requested again.
            _ = cli.sanity(self.root)
        # Then the guard runs before raw evaluation or evidence replacement.
        evaluator.assert_not_called()
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    _ = unittest.main()
