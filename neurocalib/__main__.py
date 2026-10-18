"""Fix CPU thread policy before importing the scientific runtime."""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

from neurocalib.cli import main  # noqa: E402

raise SystemExit(main())
