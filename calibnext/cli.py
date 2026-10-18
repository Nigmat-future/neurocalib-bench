"""Commands for an auditable continuation of the failed calibration search."""

# Printing is the command interface.
# ruff: noqa: T201

import argparse
import sys
from pathlib import Path
from typing import Literal, assert_never

from pydantic import ValidationError

from calib15.records import Gate
from calib15.records import read_events as parent_events
from calibnext.evaluation import evaluate_reference
from calibnext.records import read_batch, results_dir
from calibnext.schema import Sanity, Trial
from calibnext.seal import final, reproduce, same_rows
from calibnext.workflow import freeze, gate, run_trial
from neurocalib.ledger import Record, ResearchError, write_record


class Arguments(Record):
    """Validate the command boundary once, using the existing standard library."""

    root: Path
    command: Literal["sanity", "batch", "gate", "freeze", "final", "reproduce"]
    batch_id: str = ""


def sanity(root: Path) -> Path:
    """Recompute the exact old baseline using the new streaming implementation."""
    results = root / "studies" / "calib15" / "results"
    decision = Gate.model_validate_json((results / "decision.json").read_bytes())
    baseline = next(
        event
        for event in parent_events(results / "experiments.jsonl")
        if event.config == decision.baseline_best and event.score is not None
    )
    rows = evaluate_reference(root, baseline.config)
    matches = same_rows(baseline.rows, rows)
    path = results_dir(root) / "sanity.json"
    write_record(
        path, Sanity(matches_original=matches, parent_trial_id=baseline.trial_id, rows=rows)
    )
    if not matches:
        message = "Inherited baseline differs in the continued evaluator"
        raise ResearchError(message)
    return path


def _execute(args: Arguments) -> None:
    root = args.root.resolve()
    match args.command:
        case "sanity":
            print(sanity(root))
        case "batch":
            batch = read_batch(root, args.batch_id)
            for config in batch.configs:
                path = run_trial(root, config, batch.rationale, batch.batch_id)
                trial = Trial.model_validate_json(path.read_bytes())
                decision = gate(root)
                print(trial.model_dump_json(exclude={"inputs", "environment", "rows"}), flush=True)
                print(decision.model_dump_json(), flush=True)
                if decision.passed:
                    break
        case "gate":
            print(gate(root).model_dump_json(indent=2))
        case "freeze":
            print(freeze(root))
        case "final":
            print(final(root))
        case "reproduce":
            if (results_dir(root) / "final" / "report.json").exists():
                print(reproduce(root))
            else:
                from calibnext.replay import reproduce_development  # noqa: PLC0415

                print(reproduce_development(root))
        case unreachable:
            assert_never(unreachable)


def main() -> int:
    """Run one authorized research step without suppressing failures."""
    parser = argparse.ArgumentParser(description="Continued, gated fifteen-second calibration")
    _ = parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    batch = commands.add_parser("batch")
    _ = batch.add_argument("batch_id")
    for command in ("sanity", "gate", "freeze", "final", "reproduce"):
        _ = commands.add_parser(command)
    try:
        _execute(Arguments.model_validate(vars(parser.parse_args())))
    except (ResearchError, ValidationError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
