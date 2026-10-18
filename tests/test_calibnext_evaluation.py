"""The continued search preserves the information and final-access boundaries."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from calib15.schema import Features, Recording
from calibnext.evaluation import _score_day, evaluate
from calibnext.models import adapt, prepare
from calibnext.schema import Config
from neurocalib.ledger import ResearchError
from neurocalib.schema import Session


class ContinuedEvaluationChecks(unittest.TestCase):
    def test_unfrozen_final_is_rejected_before_any_recording_access(self):
        # Given a final-access guard and a recording loader that must stay unused.
        with (
            patch("calibnext.records.require_final_access", side_effect=ResearchError("sealed")),
            patch("calibnext.evaluation.load_session") as loader,
            self.assertRaisesRegex(ResearchError, "sealed"),
        ):
            # When an unsealed final evaluation is requested.
            evaluate(Path("unused"), Config(mode="map"), "test", (0, 5, 15, 30))
        # Then the guard runs before source or target data access.
        loader.assert_not_called()

    def test_scoring_labels_never_enter_prefix_adapter(self):
        # Given a valid prefix followed by labels one thousand times larger.
        times = np.arange(1, 1801) * 0.02
        x = np.column_stack([np.sin(times), np.cos(times)])
        labels = x * 2
        labels[times > 15] *= 1000
        session = Session("synthetic", times, x, labels, ("M1 001", "M1 002"))
        prepared = prepare(x[:500], x[:500] * 2, 0.1)
        spec = Recording(
            name="indy_20161011_03.mat", animal="indy", role="development", size=1, md5="0" * 32
        )
        with patch("calibnext.evaluation.adapt", wraps=adapt) as adapter:
            # When the streaming evaluator adapts and scores the same recording.
            rows = _score_day(
                spec,
                session,
                session.channels,
                prepared,
                Config(mode="map"),
                (15,),
                Features(lags=1),
            )
        # Then only the original finite prefix reaches the fitting procedure.
        self.assertEqual(len(adapter.call_args.args[2]), 750)
        self.assertLessEqual(float(np.abs(adapter.call_args.args[2]).max()), 2)
        self.assertEqual(rows[0].n_scoring_bins, int((times >= 30.5).sum()))
