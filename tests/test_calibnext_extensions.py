"""Historical serialization and per-method provenance without recording access."""

import hashlib
import json
import unittest
from pathlib import Path
from typing import override
from unittest.mock import patch

from calib15.schema import Row
from calibnext import replay, workflow
from calibnext.records import fingerprints, read_events, read_frozen
from calibnext.schema import Batch, Config, Frozen, Trial
from neurocalib.ledger import ResearchError, digest, file_digest, write_record
from tests import test_calibnext_workflow as fixture


class ExtensionChecks(unittest.TestCase):
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

    def method(self, name: str, alpha: float = 2.0) -> Config:
        directory = self.root / "methods"
        directory.mkdir(exist_ok=True)
        _ = (directory / f"{name}.py").write_text("VALUE = 1\n", encoding="utf-8")
        return Config.model_validate({"mode": "custom", "method": name, "alpha": alpha})

    def attempt(self, config: Config, batch: str = "001-custom") -> Trial:
        path = workflow.run_trial(self.root, config, "Synthetic provenance check", batch)
        return Trial.model_validate_json(path.read_bytes())

    def batch(self, name: str, configs: tuple[Config, ...], parent: str = "") -> None:
        write_record(
            self.fixture.study / "batches" / f"{name}.json",
            Batch(
                batch_id=name,
                configs=configs,
                rationale="Finite methods",
                parent_ledger_sha256=parent,
            ),
        )

    def evaluate(
        self,
        root: Path,
        config: Config,
        phase: str = "development",
        budgets: tuple[int, ...] = (15,),
    ) -> tuple[Row, ...]:
        self.assertEqual((root, phase, budgets), (self.root, "development", (15,)))
        return self.fixture.rows(0.38 if config.alpha == 1 else 0.34)

    def test_all_archived_nested_payloads_and_ids_remain_exact(self) -> None:
        # Given: both immutable archives, containing 38 starts and 38 terminal events.
        studies = Path(__file__).resolve().parents[1] / "studies"
        count = 0
        for name in ("calibnext-map-initial", "calibnext-linear-encoding"):
            lines = (
                (studies / name / "results/experiments.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            for line in lines:
                raw, trial = json.loads(line), Trial.model_validate_json(line)
                # When: the extended schema parses and serializes historical evidence.
                payload = trial.model_dump(mode="json")
                checksum = hashlib.sha256(
                    json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                # Then: neither Config identity nor any enclosing trial content changes.
                self.assertEqual(payload, raw)
                self.assertEqual(digest(trial.config), raw["trial_id"])
                self.assertEqual(digest(trial), checksum)
                self.assertEqual(len(payload["config"]), 6)
                self.assertEqual(
                    trial.model_dump(mode="json", exclude={"rows", "score"}),
                    {key: value for key, value in raw.items() if key not in {"rows", "score"}},
                )
                explicit = Config.model_validate(raw["config"] | {"method": ""})
                self.assertEqual(digest(explicit), raw["trial_id"])
                count += 1
        self.assertEqual(count, 76)

    def test_custom_identity_is_required_and_cannot_alias_a_legacy_mode(self) -> None:
        # Given: invalid leaf names and legacy configurations with a fake extension name.
        for name in ("", "../escape", "a.b", "_private", "A", "a" * 49):
            # When/Then: no invalid custom identity can be parsed.
            with self.subTest(name=name), self.assertRaises(ValueError):
                _ = Config.model_validate({"mode": "custom", "method": name})
        for mode in ("map", "pls", "encoding"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                _ = Config.model_validate({"mode": mode, "method": "fake", "bias_penalty": 0})
        self.assertEqual(self.method("valid_name").method, "valid_name")

    def test_new_method_does_not_rebind_an_earlier_trial(self) -> None:
        # Given: one completed failed method, with its own bytes recorded.
        first = self.method("first")
        self.batch("001-custom", (first,))
        with patch("calibnext.evaluation.evaluate", side_effect=self.evaluate):
            earlier = self.attempt(first)
            second = self.method("second", 3.0)
            self.batch(
                "002-custom", (second,), file_digest(self.fixture.results / "experiments.jsonl")
            )
            # When: a newly appended method runs without changing the fixed core.
            later = self.attempt(second, "002-custom")
        # Then: each trial binds exactly its own method, and the earlier trial stays valid.
        self.assertEqual(earlier.inputs, fingerprints(self.root, (first,)))
        self.assertEqual(later.inputs, fingerprints(self.root, (second,)))
        self.assertNotIn("methods/second.py", earlier.inputs)
        self.assertNotIn("methods/first.py", later.inputs)

    def test_changed_used_method_blocks_later_trial_before_evaluation(self) -> None:
        # Given: a completed first failure followed by a still-unrun method.
        first, second = self.method("first"), self.method("second", 3.0)
        self.batch("001-custom", (first, second))
        with patch("calibnext.evaluation.evaluate", side_effect=self.evaluate):
            _ = self.attempt(first)
        _ = (self.root / "methods/first.py").write_text("VALUE = 2\n", encoding="utf-8")
        with (
            patch("calibnext.evaluation.evaluate", side_effect=self.evaluate) as evaluator,
            self.assertRaisesRegex(ResearchError, "changed"),
        ):
            # When: a later trial tries to continue under edited earlier method bytes.
            _ = self.attempt(second)
        # Then: no additional label exposure or ledger start is recorded.
        evaluator.assert_not_called()
        self.assertEqual(len(read_events(self.root)), 2)

    def test_freeze_binds_failed_and_crashed_methods_but_ignores_unused_files(self) -> None:
        # Given: one failed method, one crash, and the first passing method.
        first, crashed, winner = (
            self.method("first"),
            self.method("crashed", 3),
            self.method("win", 1),
        )
        self.batch("001-custom", (first, crashed, winner))
        with patch("calibnext.evaluation.evaluate", side_effect=self.evaluate):
            _ = self.attempt(first)
        with (
            patch("calibnext.evaluation.evaluate", side_effect=RuntimeError("synthetic crash")),
            self.assertRaisesRegex(RuntimeError, "synthetic crash"),
        ):
            _ = self.attempt(crashed)
        with patch("calibnext.evaluation.evaluate", side_effect=self.evaluate):
            _ = self.attempt(winner)
        # When: the complete search is frozen and an unused method is appended.
        frozen = Frozen.model_validate_json(workflow.freeze(self.root).read_bytes())
        _ = self.method("unused")
        self.assertEqual(read_frozen(self.root), frozen)
        # Then: every exposed method remains bound, including losers and crashes.
        self.assertEqual(
            {key for key in frozen.inputs if key.startswith("methods/")},
            {"methods/first.py", "methods/crashed.py", "methods/win.py"},
        )
        for name in ("first", "crashed"):
            path = self.root / "methods" / f"{name}.py"
            original = path.read_bytes()
            _ = path.write_text("VALUE = 2\n", encoding="utf-8")
            with self.subTest(name=name), self.assertRaisesRegex(ResearchError, "changed"):
                _ = read_frozen(self.root)
            _ = path.write_bytes(original)

    def test_replay_method_edit_during_evaluation_cannot_publish_success(self) -> None:
        # Given: a completed custom trial with an unchanged numerical replay.
        config = self.method("first")
        self.batch("001-custom", (config,))
        with patch("calibnext.evaluation.evaluate", side_effect=self.evaluate):
            _ = self.attempt(config)

        def changed(root: Path, item: Config) -> tuple[Row, ...]:
            _ = (self.root / "methods/first.py").write_text("VALUE = 2\n", encoding="utf-8")
            return self.evaluate(root, item)

        with (
            patch("calibnext.replay.evaluate", side_effect=changed),
            self.assertRaisesRegex(ResearchError, "changed"),
        ):
            # When: the bound method changes during replay despite identical scores.
            _ = replay.reproduce_development(self.root)
        # Then: successful reproduction evidence is not published.
        self.assertFalse((self.fixture.results / "reproduced/development.json").exists())

    def test_replay_rejects_edited_crashed_method_before_repeating_completed_trials(self) -> None:
        # Given: a completed failure and a crash whose own method is later edited.
        first, crashed = self.method("first"), self.method("crashed", 3)
        self.batch("001-custom", (first, crashed))
        with patch("calibnext.evaluation.evaluate", side_effect=self.evaluate):
            _ = self.attempt(first)
        with (
            patch("calibnext.evaluation.evaluate", side_effect=RuntimeError("synthetic crash")),
            self.assertRaisesRegex(RuntimeError, "synthetic crash"),
        ):
            _ = self.attempt(crashed)
        _ = (self.root / "methods/crashed.py").write_text("VALUE = 2\n", encoding="utf-8")
        with (
            patch("calibnext.replay.evaluate", side_effect=self.evaluate) as evaluator,
            self.assertRaisesRegex(ResearchError, "changed"),
        ):
            # When: only completed trials would otherwise be selected for reproduction.
            _ = replay.reproduce_development(self.root)
        # Then: the crash's original input binding still blocks any numerical replay.
        evaluator.assert_not_called()
        self.assertFalse((self.fixture.results / "reproduced/development.json").exists())


if __name__ == "__main__":
    _ = unittest.main()
