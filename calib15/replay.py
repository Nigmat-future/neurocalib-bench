"""Reproduce completed development trials even when the final gate never opened."""

from pathlib import Path

from calib15.evaluation import evaluate
from calib15.io import read_study, study_dir
from calib15.records import (
    Run,
    Trial,
    fingerprints,
    read_events,
    read_features,
    validate_rows,
)
from calib15.schema import Calibration, Features, Row
from calib15.seal import scientific_equal
from neurocalib.ledger import (
    Record,
    ResearchError,
    environment,
    exclusive_run,
    file_digest,
    write_record,
)


class DevelopmentRun(Record):
    """Recomputed scientific outputs for one immutable development hypothesis."""

    trial_id: str
    config: Calibration
    features: Features
    matches_original: bool
    rows: tuple[Row, ...]


class DevelopmentReplay(Record):
    """Complete replay evidence without a final-test marker or test data access."""

    inputs: dict[str, str]
    ledger_sha256: str
    environment: dict[str, str]
    original_crashes: int
    matches_original: bool
    runs: tuple[DevelopmentRun, ...]


def reproduce_development(root: Path) -> Path:
    """Re-evaluate recorded configurations only; neither select nor add a trial."""
    with exclusive_run(study_dir(root)) as results:
        ledger = results / "experiments.jsonl"
        events = read_events(ledger)
        successful: tuple[Trial, ...] = tuple(event for event in events if event.score is not None)
        if not successful:
            message = "Development reproduction requires at least one completed trial"
            raise ResearchError(message)
        inputs, runtime, ledger_hash = fingerprints(root), environment(), file_digest(ledger)
        if any(event.inputs != inputs for event in events):
            message = "Immutable inputs changed before development reproduction"
            raise ResearchError(message)
        if any(
            runtime[name] != event.environment[name]
            for event in events
            for name in ("numpy", "h5py", "pydantic")
        ):
            message = "Development reproduction requires locked scientific dependency versions"
            raise ResearchError(message)
        features, study = read_features(root), read_study(root)
        runs: list[DevelopmentRun] = []
        for event in successful:
            if event.features != features:
                message = "Recorded trial features differ from the source-only choice"
                raise ResearchError(message)
            rows = validate_rows(
                root,
                evaluate(root, features, event.config, "development"),
                "development",
                (study.primary_budget_seconds,),
            )
            before = Run(
                label=event.trial_id,
                config=event.config,
                frozen_sha256=ledger_hash,
                rows=event.rows,
            )
            repeated = before.model_copy(update={"rows": rows})
            runs.append(
                DevelopmentRun(
                    trial_id=event.trial_id,
                    config=event.config,
                    features=features,
                    rows=rows,
                    matches_original=scientific_equal(before, repeated),
                )
            )
        if inputs != fingerprints(root) or ledger_hash != file_digest(ledger):
            message = "Inputs or ledger changed during development reproduction"
            raise ResearchError(message)
        matches = all(run.matches_original for run in runs)
        output = results / "reproduced" / "development.json"
        write_record(
            output,
            DevelopmentReplay(
                inputs=inputs,
                ledger_sha256=ledger_hash,
                environment=runtime,
                original_crashes=sum(event.status == "crash" for event in events),
                matches_original=matches,
                runs=tuple(runs),
            ),
        )
        if not matches:
            message = f"Development reproduction differs; retained evidence at {output}"
            raise ResearchError(message)
        return output
