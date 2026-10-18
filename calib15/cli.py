"""Command-line access to the independent, gated research lifecycle."""

# Printing is the command-line interface, not library-side logging.
# ruff: noqa: T201

import argparse
import sys
from pathlib import Path
from typing import Literal, assert_never

from pydantic import ValidationError

from calib15.evaluation import tune_source
from calib15.io import download_recording, read_study, results_dir
from calib15.research import final, freeze, gate, reproduce, run_trial
from calib15.schema import Calibration, Record
from neurocalib.data import DataError
from neurocalib.ledger import ResearchError


class Arguments(Record):
    """Parse argparse's untyped boundary once."""

    root: Path
    command: Literal["download", "prepare", "trial", "gate", "freeze", "final", "reproduce"]
    phase: Literal["development", "all"] = "development"
    config: Path | None = None
    hypothesis: str = ""
    kind: Literal["baseline", "candidate"] = "candidate"


def _download(root: Path, phase: str) -> None:
    for spec in read_study(root).files:
        if phase == "all" or (spec.animal == "indy" and spec.role in ("source", "development")):
            path = download_recording(root, spec)
            print(f"verified {path.name}: {spec.size} bytes", flush=True)


def _execute(args: Arguments) -> None:
    root = args.root.resolve()
    match args.command:
        case "download":
            _download(root, args.phase)
        case "prepare":
            if (results_dir(root) / "experiments.jsonl").exists():
                message = "source preparation is immutable after the first target trial"
                raise ResearchError(message)
            print(tune_source(root).model_dump_json(indent=2))
        case "trial":
            if args.config is None:
                message = "trial requires a candidate configuration"
                raise ResearchError(message)
            config = Calibration.model_validate_json(args.config.read_bytes())
            print(run_trial(root, config, args.hypothesis, args.kind))
        case "gate":
            print(gate(root).model_dump_json(indent=2))
        case "freeze":
            print(freeze(root))
        case "final":
            print(final(root))
        case "reproduce":
            print(reproduce(root))
        case unreachable:
            assert_never(unreachable)


def main() -> int:
    """Execute one action, retaining evidence and actionable failures."""
    parser = argparse.ArgumentParser(description="Gated fifteen-second supervised calibration")
    _ = parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="download and verify pinned recording bytes")
    _ = download.add_argument("--phase", choices=("development", "all"), default="development")
    _ = commands.add_parser("prepare", help="select shared features using Indy sources only")
    trial = commands.add_parser("trial", help="run one immutable development hypothesis")
    _ = trial.add_argument("--config", type=Path, required=True)
    _ = trial.add_argument("--hypothesis", required=True)
    _ = trial.add_argument("--kind", choices=("baseline", "candidate"), default="candidate")
    for command in ("gate", "freeze", "final", "reproduce"):
        _ = commands.add_parser(command)
    try:
        _execute(Arguments.model_validate(vars(parser.parse_args())))
    except (DataError, ResearchError, ValidationError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
