"""Small real HDF5 fixtures lock down the recording boundary."""

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from neurocalib.data import download_file, load_session
from neurocalib.schema import FileSpec


def write_recording(path: Path, *, gap: bool = False) -> None:
    """Write MATLAB-style transposed cell references without waveform data."""
    times = np.arange(11, dtype=np.float64) * 0.004 + 100
    if gap:
        times[5:] += 0.04
    with h5py.File(path, "w") as recording:
        recording["t"] = times.reshape(1, -1)
        recording["cursor_pos"] = np.vstack((times - 100, 2 * (times - 100)))
        names = recording.create_dataset("chan_names", (1, 3), dtype=h5py.ref_dtype)
        spikes = recording.create_dataset("spikes", (2, 3), dtype=h5py.ref_dtype)
        for column, name in enumerate(("M1 002", "S1 001", "M1 001")):
            names[0, column] = recording.create_dataset(
                f"name{column}", data=np.array([ord(letter) for letter in name], dtype=np.uint16)
            ).ref
            for unit in range(2):
                events = [100.0, 100.02, 100.04, 100.041] if unit == 0 else [100.021]
                spikes[unit, column] = recording.create_dataset(
                    f"spike{column}_{unit}", data=np.array(events).reshape(1, -1)
                ).ref


class DataTests(unittest.TestCase):
    def test_loads_m1_units_right_closed_with_causal_velocity(self):
        # Given a MATLAB recording containing mixed M1/S1 channels and two units.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indy_20161005_06.mat"
            write_recording(path)
            # When the recording is loaded into 20 ms bins.
            session = load_session(path)
        # Then units are summed per M1 channel and boundary spikes stay in the past bin.
        self.assertEqual(session.channels, ("M1 001", "M1 002"))
        np.testing.assert_allclose(session.times, [0.02, 0.04])
        np.testing.assert_array_equal(session.counts, [[1, 1], [2, 2]])
        np.testing.assert_allclose(session.velocity, [[1, 2], [1, 2]])

    def test_missing_behavior_does_not_create_a_bridge(self):
        # Given a missing 40 ms interval in the behavior stream.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indy_20161005_06.mat"
            write_recording(path, gap=True)
            # When loading the data.
            session = load_session(path)
        # Then only supported bins survive and original relative time remains intact.
        np.testing.assert_allclose(session.times, [0.08])
        np.testing.assert_allclose(session.velocity, [[1, 2]])

    def test_corrupt_download_is_not_published(self):
        # Given trusted metadata but corrupted bytes from the remote endpoint.
        spec = FileSpec(
            name="indy_20161005_06.mat",
            role="source",
            size=4,
            md5=hashlib.md5(b"good", usedforsecurity=False).hexdigest(),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("neurocalib.data.urlopen", return_value=io.BytesIO(b"evil")),
        ):
            # When fetching the recording.
            with self.assertRaisesRegex(ValueError, "checksum"):
                download_file(spec, Path(directory))
            # Then no file is accepted as the pinned recording.
            self.assertFalse((Path(directory) / spec.name).exists())

    def test_future_behavior_cannot_change_past_velocity(self):
        # Given a recording whose later cursor samples have been changed.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indy_20161005_06.mat"
            write_recording(path)
            with h5py.File(path, "r+") as recording:
                recording["cursor_pos"][:, 6:] += 10000
            # When loading bins ending before the changed samples.
            session = load_session(path)
        # Then the first velocity remains its past-only backward difference.
        np.testing.assert_allclose(session.velocity[0], [1, 2])

    def test_missing_behavior_does_not_filter_neural_prefix(self):
        # Given a cursor NaN at a valid clock timestamp.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indy_20161005_06.mat"
            write_recording(path)
            with h5py.File(path, "r+") as recording:
                recording["cursor_pos"][:, 5] = np.nan
            # When loading an unlabeled calibration prefix.
            session = load_session(path)
        # Then neural rows remain, while unavailable targets remain marked as NaN.
        np.testing.assert_allclose(session.times, [0.02, 0.04])
        np.testing.assert_array_equal(session.counts, [[1, 1], [2, 2]])
        self.assertTrue(np.isnan(session.velocity).any())


if __name__ == "__main__":
    unittest.main()
