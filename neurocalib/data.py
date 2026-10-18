"""Verified public downloads and causal M1 recording ingestion."""

import hashlib
import math
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.request import urlopen

import h5py
import numpy as np

from neurocalib.schema import FileSpec, FloatArray, Session

if TYPE_CHECKING:
    from http.client import HTTPResponse

    from numpy.typing import NDArray

DATA_URL: Final = "https://zenodo.org/records/3854034/files/"


class DataError(ValueError):
    """Recording contract failure; exception traceback metadata must remain mutable."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path: Path = path
        self.reason: str = reason
        super().__init__(f"{path.name}: {reason}")


def _verify(path: Path, spec: FileSpec) -> None:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "md5").hexdigest()
    if path.stat().st_size != spec.size or digest != spec.md5:
        raise DataError(path, "size or MD5 checksum does not match the pinned recording")


def download_file(spec: FileSpec, raw_dir: Path) -> Path:
    """Download one pinned file without publishing partial or corrupt bytes."""
    root = raw_dir.resolve()
    if re.fullmatch(r"indy_\d{8}_\d{2}\.mat", spec.name) is None:
        raise DataError(raw_dir, "unsafe recording filename")
    target = (root / spec.name).resolve()
    if target.parent != root:
        raise DataError(target, "recording path escapes the raw directory")
    root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        _verify(target, spec)
        return target
    partial: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=root, prefix=f"{spec.name}.", suffix=".partial", delete=False
        ) as output:
            partial = Path(output.name)
            # ponytail: one pinned HTTPS endpoint needs no separate HTTP dependency.
            response: HTTPResponse
            with urlopen(f"{DATA_URL}{spec.name}?download=1", timeout=60) as response:  # noqa: S310
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > spec.size:
                        raise DataError(target, "download exceeds the pinned file size")
                    _ = output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        _verify(partial, spec)
        _ = partial.replace(target)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
    return target


def _dataset(value: h5py.Dataset | h5py.Group | h5py.Datatype, path: Path) -> h5py.Dataset:
    if not isinstance(value, h5py.Dataset):
        raise DataError(path, "expected a MATLAB numeric dataset")
    return value


def _m1_counts(recording: h5py.File, edges: FloatArray) -> tuple[FloatArray, tuple[str, ...]]:
    path = Path(recording.filename)
    names = np.asarray(_dataset(recording["chan_names"], path)).reshape(-1)
    selected: list[tuple[str, int]] = []
    for index, reference in enumerate(names):
        letters = np.asarray(_dataset(recording[reference], path)).reshape(-1)
        name = "".join(chr(int(letter)) for letter in letters)
        match = re.fullmatch(r"M1\s+(\d+)", name)
        if match is not None:
            selected.append((f"M1 {int(match[1]):03d}", index))
    selected.sort()
    channels = tuple(name for name, _ in selected)
    if not channels or len(set(channels)) != len(channels):
        raise DataError(path, "expected distinct named M1 recording channels")
    references = np.asarray(_dataset(recording["spikes"], path))
    if references.shape[1:] != (names.size,):
        raise DataError(path, "spike cells must have MATLAB unit-by-channel dimensions")
    n_bins = edges.size - 1
    counts: FloatArray = np.zeros((n_bins, len(channels)), dtype=np.float64)
    for column, (_, raw_column) in enumerate(selected):
        for reference in references[:, raw_column]:
            if not reference:
                continue
            events = _dataset(recording[reference], path)
            if events.attrs.get("MATLAB_empty", 0):
                continue
            spike_times: FloatArray = np.asarray(events, dtype=np.float64).reshape(-1)
            if not np.isfinite(spike_times).all():
                raise DataError(path, "spike timestamps must be finite")
            # Right-closed bins use only spikes recorded by the current bin end.
            bins = np.searchsorted(edges, spike_times, side="left") - 1
            bins = bins[(bins >= 0) & (bins < n_bins)]
            counts[:, column] += np.bincount(bins, minlength=n_bins)
    return counts, channels


def load_session(path: Path, bin_seconds: float = 0.02) -> Session:
    """Read a MATLAB recording into causal counts and cursor velocities."""
    if not math.isfinite(bin_seconds) or bin_seconds <= 0:
        raise DataError(path, "bin_seconds must be finite and positive")
    with h5py.File(str(path), "r") as recording:
        times: FloatArray = np.asarray(_dataset(recording["t"], path), dtype=np.float64).reshape(-1)
        cursor: FloatArray = np.asarray(_dataset(recording["cursor_pos"], path), dtype=np.float64).T
        if times.size <= 1 or cursor.shape != (times.size, 2):
            raise DataError(path, "expected timestamps and matching 2-D cursor positions")
        steps: FloatArray = np.diff(times)
        if not np.isfinite(times).all() or np.any(steps <= 0):
            raise DataError(path, "timestamps must be finite and strictly increasing")
        sample_step = float(np.median(steps))
        if bin_seconds < sample_step:
            raise DataError(path, "bin width must not be smaller than the behavior sample interval")
        origin = float(times[0])
        n_bins = math.floor((float(times[-1]) - origin + bin_seconds * 1e-9) / bin_seconds)
        edges: FloatArray = origin + np.arange(n_bins + 1, dtype=np.float64) * bin_seconds
        endpoints = np.searchsorted(times, edges, side="right") - 1
        gap_step = steps > sample_step * 1.5
        bad_prefix: NDArray[np.int64] = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(gap_step, dtype=np.int64),
            )
        )
        endpoint_valid = edges - times[endpoints] <= sample_step * 1.5
        inside_gap = edges - times[endpoints] > sample_step * 1e-6
        gap_after = np.concatenate((gap_step, np.array([False])))
        endpoint_valid &= ~(inside_gap & gap_after[endpoints])
        valid = endpoint_valid[:-1] & endpoint_valid[1:]
        valid &= bad_prefix[endpoints[1:]] == bad_prefix[endpoints[:-1]]
        elapsed = times[endpoints[1:]] - times[endpoints[:-1]]
        valid &= elapsed > 0
        velocity: FloatArray = np.divide(
            np.diff(cursor[endpoints], axis=0),
            elapsed[:, None],
            out=np.full((n_bins, 2), np.nan),
            where=elapsed[:, None] > 0,
        )
        counts, channels = _m1_counts(recording, edges)
    return Session(
        name=path.stem,
        times=(edges[1:] - times[0])[valid],
        counts=counts[valid],
        velocity=velocity[valid],
        channels=channels,
    )
