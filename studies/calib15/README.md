# Initial calibration-range diagnostic

This first protocol version completed 18 source-only settings and all eight
predeclared development baselines. No candidate was evaluated and no holdout was
opened. The highest source penalty (10) and highest residual calibration penalty
(10) both won at their search boundaries. Full residual ridge rose monotonically
from mean R2 -0.878709 at alpha 0.01 to 0.332271 at alpha 10; target-only ridge
reached 0.197842 at alpha 10. Frozen source mean R2 was 0.283133.

The baseline range is not sufficient evidence of a strongly tuned comparison.
This run is retained as a range diagnostic and will not be mixed into the amended
study's ledger. The next protocol broadens source/calibration penalties and tunes
residual update strength before any candidate selection. Splits, 15-second budget,
0.02 advancement threshold and unopened holdout dates remain the same.

To reproduce this historical snapshot, check out the commit that introduced this
file and run `python -m calib15 download` followed by `python -m calib15 reproduce`.
The later archive directory is evidence; the original commit restores the exact
source code and original `studies/calib15` path used by its recorded fingerprints.
