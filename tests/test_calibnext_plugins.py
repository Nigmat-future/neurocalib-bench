"""Versioned candidate scripts use frozen inputs and cannot hide helper imports."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from calibnext.models import Prepared, prepare
from calibnext.plugins import calibrate, load_method, protect
from calibnext.schema import Config
from neurocalib.core import Decoder
from neurocalib.ledger import ResearchError
from neurocalib.schema import FloatArray


class PluginChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.methods = self.root / "methods"
        self.methods.mkdir()
        rng = np.random.default_rng(61)
        self.x = rng.normal(size=(100, 20))
        self.y = rng.normal(size=(100, 2))
        self.source = prepare(self.x, self.y, 10)

    def test_loading_new_bytes_never_reuses_python_bytecode_cache(self):
        # Given an unchanged path and two equally sized script versions.
        path = self.methods / "example.py"
        first = "def adapt(p,x,y,c):\n    return p.model\n"
        second = "def adapt(p,x,y,c):\n    return c.alpha\n"
        self.assertEqual(len(first), len(second))
        path.write_text(first, encoding="utf-8")
        config = Config(mode="custom", method="example")
        before = load_method(self.root, config)
        # When new source is written and loaded immediately.
        path.write_text(second, encoding="utf-8")
        after = load_method(self.root, config)
        # Then each callable reflects its actual source, without a stale pyc.
        self.assertIs(before(self.source, self.x, self.y, config), self.source.model)
        self.assertEqual(after(self.source, self.x, self.y, config), config.alpha)

    def test_unbound_helper_import_is_rejected_before_execution(self):
        # Given a script importing an untracked helper module.
        (self.methods / "example.py").write_text("import helper\n", encoding="utf-8")
        # When the candidate is loaded.
        with self.assertRaisesRegex(ResearchError, "single-file"):
            load_method(self.root, Config(mode="custom", method="example"))

    def test_source_and_prefix_arrays_cannot_be_mutated(self):
        # Given immutable cached source statistics and source-compatible calibration.
        protect(self.source)
        adapter = load_method(self.root, Config(mode="map"))
        original = self.source.model.weights.copy()
        # When the ordinary calibrator runs with protected observations.
        result = calibrate(adapter, self.source, self.x, self.y, Config(mode="map"))
        # Then source stays unchanged and all shared input arrays are read-only.
        self.assertFalse(self.source.model.weights.flags.writeable)
        encoding = self.source.encoding
        assert encoding is not None
        self.assertFalse(encoding.precision.flags.writeable)
        self.assertFalse(self.x.flags.writeable)
        self.assertTrue(np.isfinite(result.weights).all())
        np.testing.assert_array_equal(self.source.model.weights, original)

    def test_nonfinite_candidate_readout_is_rejected(self):
        # Given a candidate returning invalid finite-shape parameters.
        def invalid(prepared: Prepared, _x: FloatArray, _y: FloatArray, _config: Config) -> Decoder:
            model = prepared.model
            return Decoder(model.center, model.scale, model.weights * np.nan, model.bias)

        # When its output crosses the scoring boundary.
        with self.assertRaisesRegex(ResearchError, "finite"):
            calibrate(invalid, self.source, self.x, self.y, Config(mode="map"))
