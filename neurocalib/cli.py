"""Small standard-library command line for the complete experiment lifecycle."""

import argparse
import csv
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from neurocalib.data import DataError, download_file
from neurocalib.evaluation import evaluate, read_protocol
from neurocalib.ledger import ResearchError
from neurocalib.research import final_test, freeze, record_trial
from neurocalib.schema import Strategy


def main() -> int:
    """Parse one explicit action; preserve actionable errors and nonzero exits."""
    parser = argparse.ArgumentParser(description="Budget-controlled cross-day neural decoding")
    _ = parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    commands = parser.add_subparsers(dest="command", required=True)
    _ = commands.add_parser("download", help="download and verify all six pinned recordings")
    _ = commands.add_parser("baseline", help="score default frozen ridge on development days only")
    trial = commands.add_parser("trial", help="evaluate one AI-proposed development hypothesis")
    _ = trial.add_argument("--config", type=Path, required=True)
    _ = trial.add_argument("--hypothesis", required=True)
    _ = commands.add_parser("freeze", help="seal the development winner and comparison methods")
    _ = commands.add_parser("final", help="run or resume exactly the sealed final comparison")
    _ = commands.add_parser(
        "reproduce", help="recompute sealed results without allowing new selection"
    )
    _ = commands.add_parser("report", help="render figures from the saved final scores")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        match args.command:
            case "download":
                for spec in read_protocol(root).files:
                    path = download_file(spec, root / "data" / "raw")
                    print(f"verified {path.name}: {spec.size} bytes")
            case "baseline":
                rows = evaluate(root, Strategy(), "development")
                path = root / "results" / "baseline.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                _ = path.write_text(
                    json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
                with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                control = evaluate(root, Strategy(), "development", shuffle_source=True)
                _ = path.with_name("source_shift_control.json").write_text(
                    json.dumps(control, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
                print(path)
            case "trial":
                print(record_trial(root, args.config.resolve(), args.hypothesis))
            case "freeze":
                print(freeze(root))
            case "final":
                print(final_test(root))
            case "reproduce":
                from neurocalib.reproduce import reproduce  # noqa: PLC0415

                print(reproduce(root))
            case "report":
                from neurocalib.report import render  # noqa: PLC0415

                print(render(root))
            case _:
                parser.error("unknown command")
    except (DataError, ResearchError, ValidationError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
