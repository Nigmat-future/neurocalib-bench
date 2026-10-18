"""Synthetic seals enforce pre-read guards, complete caches and the original gate."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from calib15.schema import Calibration
from calibnext import cli, evaluation, records, seal, workflow
from calibnext.schema import Config, Sanity
from neurocalib.ledger import ResearchError, Started, file_digest, write_record
from tests import test_calibnext_workflow as fixtures


class SealChecks(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CalibNextTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.results = self.fixture.root, self.fixture.results
        protocol_path = self.fixture.parent / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["files"].extend(
            {
                "name": name,
                "animal": "loco",
                "role": role,
                "size": 1,
                "md5": "0" * 32,
            }
            for name, role in (("loco_20160101_01.mat", "source"), ("loco_20160301_01.mat", "test"))
        )
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        inherited = records.read_inheritance(self.root)
        write_record(
            self.results / "sanity.json",
            Sanity(
                matches_original=True,
                parent_trial_id=inherited.baseline.trial_id,
                rows=inherited.baseline.rows,
            ),
        )
        self.config = Config(alpha=1)

    def freeze(self):
        self.fixture.batch("001-map", (self.config,))
        with patch("calibnext.evaluation.evaluate", return_value=self.fixture.rows(0.38)):
            workflow.run_trial(self.root, self.config, "Synthetic passing candidate", "001-map")
        path = workflow.freeze(self.root)
        return file_digest(path)

    def target_rows(self, indy=0.53, loco=0.53):
        template = self.fixture.rows()[0]
        return tuple(
            template.model_copy(
                update={
                    "animal": animal,
                    "session": f"{animal}_20160301_01.mat",
                    "budget_seconds": budget,
                    "r2": score,
                    "r2_x": score,
                    "r2_y": score,
                    "frozen_r2": 0.4,
                }
            )
            for animal, score in (("indy", indy), ("loco", loco))
            for budget in (0, 5, 15, 30)
        )

    def test_unfrozen_final_stops_before_downloading_or_loading_recordings(self):
        # Given no final freeze, with both raw-data entry points observed.
        with (
            patch("calibnext.seal.download_recording") as downloader,
            patch("calibnext.evaluation.load_session") as loader,
            self.assertRaisesRegex(ResearchError, "freeze"),
        ):
            # When final validation is requested before a passing development decision.
            seal.final(self.root)
        # Then neither downloads nor behavior reads start, and no marker is created.
        downloader.assert_not_called()
        loader.assert_not_called()
        self.assertFalse((self.results / "final.started.json").exists())

    def test_sanity_cannot_overwrite_bound_evidence_after_any_trial_starts(self):
        # Given the first cumulative ledger, which already binds the sanity evidence.
        (self.results / "experiments.jsonl").touch()
        path = self.results / "sanity.json"
        before = path.read_bytes()
        with (
            patch("calibnext.cli.evaluate_reference") as evaluator,
            self.assertRaises(ResearchError),
        ):
            # When the source parity command is accidentally requested again.
            cli.sanity(self.root)
        # Then no labels are reread and no newly timed record replaces the bound file.
        evaluator.assert_not_called()
        self.assertEqual(path.read_bytes(), before)

    def test_final_marker_is_valid_before_the_first_download(self):
        # Given a real synthetic passing freeze with no final marker yet.
        self.freeze()
        observed = []

        def guarded_download(root, spec):
            records.require_final_access(root)
            observed.append(spec.name)
            message = "Stop after guarded download boundary"
            raise ResearchError(message)

        with (
            patch("calibnext.seal.download_recording", side_effect=guarded_download),
            patch("calibnext.evaluation.load_session") as loader,
            self.assertRaisesRegex(ResearchError, "guarded download"),
        ):
            # When final validation reaches its first permitted data operation.
            seal.final(self.root)
        # Then the actual seal-bound marker preceded that operation, without opening raw data.
        self.assertEqual(observed, ["indy_20160301_01.mat"])
        loader.assert_not_called()

    def test_sealed_configuration_and_full_budgets_are_required_before_any_loader(self):
        # Given an authorized marker bound to a valid synthetic freeze.
        checksum = self.freeze()
        write_record(self.results / "final.started.json", Started(frozen_sha256=checksum))
        with (
            patch("calibnext.evaluation.recording_path") as verifier,
            patch("calibnext.evaluation.load_session") as loader,
        ):
            for config, budgets in ((Config(alpha=2), (0, 5, 15, 30)), (self.config, (15,))):
                # When callers change the selected model or request a partial final comparison.
                with self.subTest(config=config, budgets=budgets), self.assertRaises(ResearchError):
                    evaluation.evaluate(self.root, config, "test", budgets)
            with self.assertRaises(ResearchError):
                evaluation.evaluate_reference(
                    self.root, Calibration(mode="residual", alpha=0.01), "test", (0, 5, 15, 30)
                )
        # Then refusal occurs before byte verification or source/target loading.
        verifier.assert_not_called()
        loader.assert_not_called()

    def test_cached_final_config_or_missing_day_budget_cannot_be_reused(self):
        # Given a synthetic freeze and a cache with either the wrong method or a missing cell.
        checksum = self.freeze()
        path = self.results / "final/winner.json"
        variants = ((Config(alpha=2), self.target_rows()), (self.config, self.target_rows()[:-1]))
        for config, rows in variants:
            write_record(
                path, seal.Run(label="winner", config=config, frozen_sha256=checksum, rows=rows)
            )
            with (
                self.subTest(config=config, cells=len(rows)),
                patch("calibnext.seal.download_recording", return_value=Path("unused")),
                patch("calibnext.seal._run") as runner,
                self.assertRaises(ResearchError),
            ):
                # When final validation attempts to reuse that saved comparison.
                seal.final(self.root)
            # Then the invalid cached result cannot enter a final outcome.
            runner.assert_not_called()
            self.assertFalse((self.results / "final/report.json").exists())

    def test_input_changes_during_download_cannot_escape_through_complete_caches(self):
        # Given a complete cache for all four genuinely sealed comparison configurations.
        checksum = self.freeze()
        frozen = records.read_frozen(self.root)
        for label, config in seal.comparisons(frozen):
            write_record(
                self.results / "final" / f"{label}.json",
                seal.Run(
                    label=label, config=config, frozen_sha256=checksum, rows=self.target_rows()
                ),
            )

        def changed_input(root, spec):
            (root / "calibnext/core.py").write_text("CORE = 2\n", encoding="utf-8")
            return Path(spec.name)

        with (
            patch("calibnext.seal.download_recording", side_effect=changed_input),
            patch("calibnext.seal._run") as runner,
            self.assertRaisesRegex(ResearchError, "changed"),
        ):
            # When a bound scientific input changes after the initial access check.
            seal.final(self.root)
        # Then cache reuse still checks the seal again before committing a final report.
        runner.assert_not_called()
        self.assertFalse((self.results / "final/report.json").exists())

    def test_final_outcome_preserves_macro_gain_and_both_animal_harm_rules(self):
        # Given fixed baseline scores and candidates spanning the advancement boundary.
        baseline = seal.Run(
            label="baseline",
            config=Calibration(mode="residual"),
            frozen_sha256="synthetic",
            rows=self.target_rows(0.5, 0.5),
        )
        cases = ((0.52, 0.52, True), (0.52, 0.5, False), (0.6, 0.49, False))
        for indy, loco, passed in cases:
            chosen = seal.Run(
                label="winner",
                config=self.config,
                frozen_sha256="synthetic",
                rows=self.target_rows(indy, loco),
            )
            # When the original 15-second, equal-animal mean criterion is applied.
            result = seal.outcome((chosen, baseline))
            # Then enough aggregate gain never excuses a degraded animal.
            self.assertEqual(result.passed, passed)
            self.assertAlmostEqual(result.macro_delta, (indy + loco) / 2 - 0.5)
        rows = list(self.target_rows(0.54, 0.54))
        harmed = rows[2].model_copy(update={"r2": 0.39})
        compensating = rows[2].model_copy(update={"session": "indy_20160302_01.mat", "r2": 0.69})
        rows[2:3] = [harmed, compensating]
        chosen = seal.Run(
            label="winner", config=self.config, frozen_sha256="synthetic", rows=tuple(rows)
        )
        baseline = baseline.model_copy(
            update={
                "rows": (
                    *baseline.rows,
                    baseline.rows[2].model_copy(update={"session": compensating.session}),
                )
            }
        )
        # When one harmed day is offset by a very successful second day.
        result = seal.outcome((chosen, baseline))
        # Then the positive mean gain still cannot override additional negative transfer.
        self.assertGreater(result.macro_delta, 0.02)
        self.assertFalse(result.passed)

    def test_numerical_replay_excludes_clock_time_but_detects_score_or_row_changes(self):
        # Given one recorded scientific result and a different machine's timing.
        original = self.target_rows()
        timing = tuple(row.model_copy(update={"calibration_seconds": 10.0}) for row in original)
        changed = list(timing)
        changed[0] = changed[0].model_copy(update={"r2": changed[0].r2 + 1e-5})
        # When comparing reproduction rows at the declared 1e-7 tolerance.
        self.assertTrue(seal.same_rows(original, timing))
        # Then scientific drift and incomplete comparisons fail independently of runtime.
        self.assertFalse(seal.same_rows(original, tuple(changed)))
        self.assertFalse(seal.same_rows(original, original[:-1]))


if __name__ == "__main__":
    unittest.main()
