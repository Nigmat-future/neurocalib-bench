"""Pin numerical threads before importing the scientific runtime."""

import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

from calib15.cli import main  # noqa: E402

raise SystemExit(main())
