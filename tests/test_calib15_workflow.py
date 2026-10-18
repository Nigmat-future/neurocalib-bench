"""Drive the real CLI against MATLAB-reference fixtures with no holdout files."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.fixtures import make_repository


class CalibrationWorkflowTests(unittest.TestCase):
    def test_cli_prepares_and_scores_without_any_holdout_bytes(self):
        # Given real tiny HDF5 sources and development files, but no sealed recordings.
        original = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_repository(root)
            manifest = json.loads(
                (original / "studies" / "calib15" / "protocol.json").read_text(encoding="utf-8")
            )
            old = sorted((root / "data" / "raw").glob("*.mat"))
            data = [path.read_bytes() for path in old]
            for index, spec in enumerate(manifest["files"]):
                if spec["animal"] == "indy" and spec["role"] != "test":
                    payload = data[index % len(data)]
                    _ = (root / "data" / "raw" / spec["name"]).write_bytes(payload)
                    spec["size"] = len(payload)
                    spec["md5"] = hashlib.md5(payload, usedforsecurity=False).hexdigest()
            study = root / "studies" / "calib15"
            study.mkdir(parents=True)
            _ = (study / "protocol.json").write_text(json.dumps(manifest), encoding="utf-8")
            for name in ("PROTOCOL.md", "program.md"):
                _ = (study / name).write_text("synthetic integration fixture\n", encoding="utf-8")
            config = root / "baseline.json"
            _ = config.write_text('{"mode":"residual","alpha":1.0}', encoding="utf-8")
            env = {**os.environ, "PYTHONPATH": str(original)}
            commands = (
                ("prepare",),
                ("trial", "--kind", "baseline", "--config", str(config), "--hypothesis", "check"),
                ("gate",),
            )
            # When the actual module command runs the permitted development lifecycle.
            outputs = [
                subprocess.run(  # noqa: S603
                    [sys.executable, "-m", "calib15", "--root", str(root), *command],
                    cwd=original,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )
                for command in commands
            ]
            # Then one scored development result and a failed gate exist without final access.
            for output in outputs:
                self.assertEqual(output.returncode, 0, output.stderr)
            decision = json.loads((study / "results" / "decision.json").read_bytes())
            self.assertFalse(decision["passed"])
            self.assertFalse((study / "results" / "final.started.json").exists())
            self.assertEqual(len(list((study / "results" / "trials").glob("*/result.json"))), 1)

    def test_cli_final_rejects_missing_freeze_before_reading_data(self):
        # Given a directory containing no data or completed development study.
        with tempfile.TemporaryDirectory() as directory:
            # When a user attempts the real final command.
            output = subprocess.run(  # noqa: S603
                [sys.executable, "-m", "calib15", "--root", directory, "final"],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
            # Then it refuses at the research boundary, before any recording access.
            self.assertNotEqual(output.returncode, 0)
            self.assertIn("passing", output.stderr)
