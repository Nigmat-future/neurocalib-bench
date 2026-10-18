"""Boundary and checksum checks for the independent study."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from calib15.io import recording_path
from calib15.schema import Recording
from neurocalib.data import DataError


class StudyBoundaryTests(unittest.TestCase):
    def test_recording_rejects_mismatched_animal(self):
        # Given a recording attributed to another animal.
        values = {"name": "indy_20161011_03.mat", "animal": "loco", "role": "test"}
        # When parsing the manifest, then the mismatch is rejected.
        with self.assertRaises(ValidationError):
            Recording.model_validate({**values, "size": 1, "md5": "0" * 32})

    def test_same_size_corruption_is_rejected(self):
        # Given bytes that have the right size but an incorrect official digest.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data" / "raw"
            raw.mkdir(parents=True)
            path = raw / "loco_20170215_02.mat"
            _ = path.write_bytes(b"wrong")
            spec = Recording(
                name=path.name,
                animal="loco",
                role="test",
                size=5,
                md5=hashlib.md5(b"right", usedforsecurity=False).hexdigest(),
            )
            # When resolving the recording, then corruption is rejected before HDF5 access.
            with self.assertRaisesRegex(DataError, "MD5"):
                recording_path(root, spec)
