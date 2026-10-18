"""Two explicit historical archives retain cumulative exposure and immutable evidence."""

import unittest
from pathlib import Path
from typing import override
from unittest.mock import patch

from calibnext import workflow
from calibnext.records import fingerprints, read_events, read_inheritance
from calibnext.schema import Batch, Config, Decision, Trial, decide
from neurocalib.ledger import ResearchError, digest, environment, write_record
from tests import test_calibnext_workflow as fixture


class AncestryChecks(unittest.TestCase):
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

    @staticmethod
    def first_configs() -> tuple[Config, ...]:
        return tuple(Config(alpha=100 + number) for number in range(8))

    @staticmethod
    def second_configs() -> tuple[Config, ...]:
        return (
            *(
                Config(mode="pls", alpha=20 + number, components=number + 1, bias_penalty=0)
                for number in range(18)
            ),
            *(Config(mode="encoding", alpha=50 + number) for number in range(12)),
        )

    def archive(self, name: str, configs: tuple[Config, ...], previous: int) -> Path:
        inherited = read_inheritance(self.root).model_copy(
            update={"previous_continuation_attempts": previous}
        )
        events: list[Trial] = []
        for config in configs:
            start = Trial(
                sequence=len(events) + 1,
                trial_id=digest(config),
                batch_id="001-archive",
                batch_sha256="0" * 64,
                config=config,
                features=self.fixture.features,
                hypothesis="Previously exposed synthetic development",
                inputs={"historical_core_no_longer_present.py": "0" * 64},
                environment=environment(),
                started_at="2000-01-01T00:00:00+00:00",
                status="running",
            )
            events.extend(
                (
                    start,
                    start.model_copy(
                        update={
                            "sequence": start.sequence + 1,
                            "status": "discard",
                            "score": 0.34,
                            "rows": self.fixture.rows(0.34),
                        }
                    ),
                )
            )
        directory = self.root / "studies" / name
        results = directory / "results"
        results.mkdir(parents=True)
        _ = (directory / "protocol.json").write_text("{}\n", encoding="utf-8")
        _ = (results / "experiments.jsonl").write_text(
            "".join(event.model_dump_json() + "\n" for event in events), encoding="utf-8"
        )
        decision = decide(tuple(events), inherited)
        _ = (results / "decision.json").write_text(
            decision.model_dump_json(
                exclude={"previous_continuation_attempts"} if not previous else {}
            )
            + "\n",
            encoding="utf-8",
        )
        _ = (results / "sanity.json").write_bytes(
            (self.fixture.results / "sanity.json").read_bytes()
        )
        return directory

    def both_archives(self) -> tuple[Path, Path]:
        first = self.archive("calibnext-map-initial", self.first_configs(), 0)
        second = self.archive("calibnext-linear-encoding", self.second_configs(), 8)
        return first, second

    def test_history_retains_zero_eight_thirty_eight_chain_without_old_code(self) -> None:
        # Given: no archives, then the original eight failed MAP attempts.
        self.assertEqual(read_inheritance(self.root).previous_continuation_attempts, 0)
        first = self.archive("calibnext-map-initial", self.first_configs(), 0)
        self.assertEqual(read_inheritance(self.root).previous_continuation_attempts, 8)
        # When: the following 30 failed PLS/encoding attempts are inherited in order.
        second = self.archive("calibnext-linear-encoding", self.second_configs(), 8)
        inherited = read_inheritance(self.root)
        # Then: exposure becomes 38, keeping the original 32 configurations/8 candidates.
        self.assertEqual(inherited.previous_continuation_attempts, 38)
        self.assertEqual((inherited.configurations, inherited.candidates), (32, 8))
        self.assertEqual(workflow.gate(self.root).previous_continuation_attempts, 38)
        self.assertEqual(len(read_events(self.root, archive=True)), 76)
        for folder, previous in ((first, 0), (second, 8)):
            decision = Decision.model_validate_json((folder / "results/decision.json").read_bytes())
            self.assertEqual(decision.previous_continuation_attempts, previous)
            for relative in (
                "protocol.json",
                "results/experiments.jsonl",
                "results/decision.json",
                "results/sanity.json",
            ):
                self.assertIn(
                    (folder / relative).relative_to(self.root).as_posix(), fingerprints(self.root)
                )

    def test_declared_prior_attempts_cannot_survive_missing_archives(self) -> None:
        # Given: current metadata declares 38 prior attempts but all archives are absent.
        _ = (self.fixture.study / "protocol.json").write_text(
            '{"previous_continuation_attempts":38}\n', encoding="utf-8"
        )
        # When/Then: archive deletion cannot silently reset inherited exposure to zero.
        with self.assertRaises(ResearchError):
            _ = read_inheritance(self.root)

    def test_declared_prior_attempts_must_match_both_complete_archives(self) -> None:
        # Given: both archives contain exactly 38 prior attempts, matching current metadata.
        _ = self.both_archives()
        protocol = self.fixture.study / "protocol.json"
        _ = protocol.write_text('{"previous_continuation_attempts":38}\n', encoding="utf-8")
        self.assertEqual(read_inheritance(self.root).previous_continuation_attempts, 38)
        # When: current metadata instead claims no prior continuation attempts.
        _ = protocol.write_text('{"previous_continuation_attempts":0}\n', encoding="utf-8")
        # Then: metadata cannot contradict the complete historical evidence.
        with self.assertRaises(ResearchError):
            _ = read_inheritance(self.root)

    def test_second_archive_cannot_erase_its_eight_prior_attempts(self) -> None:
        # Given: two otherwise complete archives with their historical decision chain.
        _, second = self.both_archives()
        path = second / "results/decision.json"
        decision = Decision.model_validate_json(path.read_bytes())
        write_record(path, decision.model_copy(update={"previous_continuation_attempts": 0}))
        # When/Then: rewriting second-stage exposure to zero is rejected.
        with self.assertRaises(ResearchError):
            _ = read_inheritance(self.root)

    def test_second_archive_requires_its_first_archive(self) -> None:
        # Given: 30 historical attempts claiming an eight-attempt predecessor that is absent.
        _ = self.archive("calibnext-linear-encoding", self.second_configs(), 8)
        # When/Then: a later archive cannot silently become the entire history.
        with self.assertRaises(ResearchError):
            _ = read_inheritance(self.root)

    def test_cross_archive_duplicate_configuration_is_rejected(self) -> None:
        # Given: a complete first archive and one repeated configuration among the next 30.
        _ = self.archive("calibnext-map-initial", self.first_configs(), 0)
        configs = (self.first_configs()[0], *self.second_configs()[1:])
        _ = self.archive("calibnext-linear-encoding", configs, 8)
        # When/Then: a repeated identity cannot count as another distinct historical attempt.
        with self.assertRaisesRegex(ResearchError, "repeated across"):
            _ = read_inheritance(self.root)

    def test_linear_encoding_archive_cannot_be_retried_in_the_current_study(self) -> None:
        # Given: both archives and a current manifest reusing the last encoding identity.
        _ = self.both_archives()
        config = self.second_configs()[-1]
        write_record(
            self.fixture.study / "batches/001-repeated.json",
            Batch(batch_id="001-repeated", configs=(config,), rationale="Repeated historical ID"),
        )
        with (
            patch("calibnext.evaluation.evaluate") as evaluator,
            self.assertRaisesRegex(ResearchError, "archiv"),
        ):
            # When: a previously exposed second-archive configuration is submitted again.
            _ = workflow.run_trial(self.root, config, "Repeated old encoding", "001-repeated")
        # Then: the old exposure is rejected before any development evaluation.
        evaluator.assert_not_called()
        self.assertFalse((self.fixture.results / "experiments.jsonl").exists())


if __name__ == "__main__":
    _ = unittest.main()
