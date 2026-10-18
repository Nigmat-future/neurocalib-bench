"""Pin scientific compute threads before importing numerical libraries."""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

from calibnext.cli import main  # noqa: E402

raise SystemExit(main())
