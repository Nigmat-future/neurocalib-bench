"""Exercise the shipped CLI against real tiny HDF5 files without network access."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.fixtures import make_repository


class WorkflowChecks(unittest.TestCase):
    def run_command(self, root, *args):
        return subprocess.run(  # noqa: S603
            [sys.executable, "-m", "neurocalib", "--root", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_development_freeze_final_reproduce_and_search_stop(self):
        # Given real MATLAB-style fixtures and one predeclared candidate.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_repository(root)
            config = root / "candidate.json"
            _ = config.write_text('{"mode":"center","lags":2}', encoding="utf-8")
            # When the CLI executes the complete scientific lifecycle.
            for args in [
                ("baseline",),
                ("trial", "--config", str(config), "--hypothesis", "offset check"),
                ("freeze",),
                ("final",),
                ("reproduce",),
                ("report",),
            ]:
                result = self.run_command(root, *args)
                self.assertEqual(result.returncode, 0, result.stderr)
            # Then frozen results are reproducible and no later search is allowed.
            report = json.loads((root / "results/final/report.json").read_text())
            self.assertEqual(len(report["results"]), 7)
            self.assertEqual(len(report["results"][0]["rows"]), 8)
            self.assertTrue((root / "results/final/calibration_curves.png").is_file())
            stopped = self.run_command(
                root, "trial", "--config", str(config), "--hypothesis", "blocked"
            )
            self.assertNotEqual(stopped.returncode, 0)
            self.assertIn("Search stopped", stopped.stderr)

    def test_invalid_candidate_exits_without_creating_a_trial(self):
        # Given an out-of-protocol history length.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_repository(root)
            config = root / "candidate.json"
            _ = config.write_text('{"mode":"center","lags":21}', encoding="utf-8")
            # When it crosses the actual CLI trust boundary.
            result = self.run_command(root, "trial", "--config", str(config), "--hypothesis", "bad")
            # Then invalid input is rejected before the experiment ledger exists.
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "results/experiments.jsonl").exists())

    def test_corrupt_recording_preserves_the_data_error_and_crash_record(self):
        # Given a source file whose content no longer matches its pinned MD5.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_repository(root)
            source = root / "data/raw/indy_20160930_05.mat"
            with source.open("ab") as stream:
                _ = stream.write(b"corrupt")
            config = root / "candidate.json"
            _ = config.write_text('{"mode":"center","lags":2}', encoding="utf-8")
            # When a trial fails inside the exclusive experiment lock.
            result = self.run_command(
                root, "trial", "--config", str(config), "--hypothesis", "bad data"
            )
            # Then the original integrity error remains actionable and is recorded.
            self.assertEqual(result.returncode, 1)
            self.assertIn("MD5 checksum", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            events = (root / "results/experiments.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(events[-1])["status"], "crash")
            self.assertFalse((root / "results/.active").exists())


if __name__ == "__main__":
    unittest.main()
