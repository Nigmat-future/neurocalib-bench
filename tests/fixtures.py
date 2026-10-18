"""Tiny real HDF5 recordings for offline integration checks, not biological evidence."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def make_repository(root: Path) -> None:
    """Create a six-day, four-channel fixture in the actual MATLAB reference format."""
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    original = Path(__file__).resolve().parents[1]
    protocol = json.loads((original / "protocol.json").read_text(encoding="utf-8"))
    for seed, spec in enumerate(protocol["files"]):
        path = raw / spec["name"]
        rng = np.random.default_rng(seed)
        times = np.arange(10001, dtype=float) / 250
        velocity = np.column_stack([np.sin(times * 2), np.cos(times * 1.5)]) * 20
        position = np.cumsum(velocity, axis=0) / 250
        with h5py.File(path, "w") as handle:
            handle.create_dataset("t", data=times[None, :])
            handle.create_dataset("cursor_pos", data=position.T)
            names = handle.create_dataset("chan_names", (1, 4), dtype=h5py.ref_dtype)
            cells = handle.create_dataset("spikes", (1, 4), dtype=h5py.ref_dtype)
            for channel in range(4):
                text = handle.create_dataset(
                    f"name{channel}",
                    data=np.array([ord(c) for c in f"M1 {channel + 1:03d}"], dtype=np.uint16),
                )
                names[0, channel] = text.ref
                events = np.sort(rng.uniform(0, 40, size=800))
                spike = handle.create_dataset(f"events{channel}", data=events[None, :])
                cells[0, channel] = spike.ref
        spec["size"] = path.stat().st_size
        spec["md5"] = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    _ = (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    for name in ("PROTOCOL.md", "pyproject.toml", "uv.lock"):
        _ = (root / name).write_text("integration fixture\n", encoding="utf-8")
    (root / "neurocalib").mkdir()
