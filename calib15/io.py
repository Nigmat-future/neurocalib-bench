"""Pinned recording access and paths, without changing the original pilot."""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlopen

from calib15.schema import Recording, Study
from neurocalib.data import DATA_URL, DataError

if TYPE_CHECKING:
    from http.client import HTTPResponse


def study_dir(root: Path) -> Path:
    """Keep this experiment independent of the original frozen results."""
    return root / "studies" / "calib15"


def results_dir(root: Path) -> Path:
    """Return the new study's evidence directory."""
    return study_dir(root) / "results"


def read_study(root: Path) -> Study:
    """Parse the complete access policy before loading recordings."""
    return Study.model_validate_json((study_dir(root) / "protocol.json").read_bytes())


def recording_path(root: Path, spec: Recording) -> Path:
    """Verify local bytes; this function never downloads or reads behavior labels."""
    raw = (root / "data" / "raw").resolve()
    target = (raw / spec.name).resolve()
    if target.parent != raw:
        raise DataError(target, "recording path escapes the raw directory")
    if target.stat().st_size != spec.size:
        raise DataError(target, "size does not match the pinned recording")
    with target.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "md5").hexdigest()
    if checksum != spec.md5:
        raise DataError(target, "MD5 does not match the pinned recording")
    return target


def download_recording(root: Path, spec: Recording) -> Path:
    """Download only the selected pinned recording and atomically publish valid bytes."""
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / spec.name
    if target.exists():
        return recording_path(root, spec)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=raw, prefix=f"{spec.name}.", suffix=".partial", delete=False
        ) as output:
            temporary = Path(output.name)
            # ponytail: one pinned HTTPS endpoint needs only the standard library.
            response: HTTPResponse
            with urlopen(f"{DATA_URL}{spec.name}?download=1", timeout=60) as response:  # noqa: S310
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > spec.size:
                        raise DataError(temporary, "download exceeds pinned size")
                    _ = output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        with temporary.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "md5").hexdigest()
        if temporary.stat().st_size != spec.size or checksum != spec.md5:
            raise DataError(temporary, "download size or MD5 differs from the pinned recording")
        _ = temporary.replace(target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
