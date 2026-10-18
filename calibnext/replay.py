"""Repeat exposed development trials without adding search or accessing holdouts."""

from pathlib import Path

from calib15.records import validate_rows
from calib15.schema import Row
from calibnext.evaluation import evaluate
from calibnext.records import (
    batch_path,
    fingerprints,
    read_events,
    read_inheritance,
    study_dir,
)
from calibnext.schema import Config
from calibnext.seal import same_rows
from neurocalib.ledger import (
    Record,
    ResearchError,
    environment,
    exclusive_run,
    file_digest,
    write_record,
)


class RepeatedTrial(Record):
    """One fixed configuration rerun against its existing daily evidence."""

    trial_id: str
    config: Config
    matches_original: bool
    rows: tuple[Row, ...]


class Reproduction(Record):
    """All successful exposed trials, with no new winner selection."""

    inputs: dict[str, str]
    ledger_sha256: str
    environment: dict[str, str]
    matches_original: bool
    trials: tuple[RepeatedTrial, ...]


def reproduce_development(root: Path) -> Path:
    """Reevaluate recorded configurations only, keeping final access sealed."""
    with exclusive_run(study_dir(root)) as results:
        _ = read_inheritance(root)
        ledger_hash = file_digest(results / "experiments.jsonl")
        events = read_events(root)
        configs = tuple(event.config for event in events)
        inputs, runtime = fingerprints(root, configs), environment()
        if file_digest(results / "experiments.jsonl") != ledger_hash:
            message = "Development ledger changed while reading the replay snapshot"
            raise ResearchError(message)
        manifests = {event.batch_id: event.batch_sha256 for event in events}
        complete = tuple(event for event in events if event.score is not None)
        if not complete:
            message = "No completed continued-development trials to reproduce"
            raise ResearchError(message)
        if any(
            event.inputs != fingerprints(root, (event.config,))
            or event.batch_sha256 != file_digest(batch_path(root, event.batch_id))
            or any(event.environment[key] != runtime[key] for key in ("numpy", "h5py", "pydantic"))
            for event in events
        ):
            message = "Reproduction inputs, manifest or scientific dependencies changed"
            raise ResearchError(message)
        trials: list[RepeatedTrial] = []
        for event in complete:
            rows = validate_rows(root, evaluate(root, event.config), "development", (15,))
            trials.append(
                RepeatedTrial(
                    trial_id=event.trial_id,
                    config=event.config,
                    matches_original=same_rows(event.rows, rows),
                    rows=rows,
                )
            )
        output = results / "reproduced" / "development.json"
        if (
            fingerprints(root, configs) != inputs
            or any(event.inputs != fingerprints(root, (event.config,)) for event in events)
            or file_digest(results / "experiments.jsonl") != ledger_hash
            or any(
                file_digest(batch_path(root, name)) != checksum
                for name, checksum in manifests.items()
            )
        ):
            message = "Research inputs changed during development reproduction"
            raise ResearchError(message)
        matches = all(trial.matches_original for trial in trials)
        write_record(
            output,
            Reproduction(
                inputs=inputs,
                ledger_sha256=ledger_hash,
                environment=runtime,
                matches_original=matches,
                trials=tuple(trials),
            ),
        )
        if not matches:
            message = "Continued development reproduction differs; evidence retained"
            raise ResearchError(message)
        return output
