# Continued fifteen-second calibration research

This continues the failed [PCA/affine study](../calib15/README.md) on the same nine
Indy development dates, fixed source features, strongest supervised baseline,
and 15-second labeled prefix. **The frozen candidate failed the prespecified
two-animal final criterion.** Its development improvement did not transfer to
both animals. The candidate and protocol are retained; selection and tuning
have ended. The original six-day pilot is a separate, already completed
experiment with its own published final results.

This version introduced calibration algorithms as separately hashed single-file
methods while keeping evaluation, labels, scoring, and the gate fixed. Its
development selection repeatedly reused the same dates; the final comparison
was evaluated only after the selection was frozen.

## Frozen final outcome: failed

The [final report](results/final/report.json) evaluates the frozen candidate and
three fixed references: the development-selected residual baseline, target-only
ridge, and frozen source ridge. Four comparisons across twelve final dates and
budgets 0, 5, 15, and 30 seconds produce **192 rows**. These method/budget cells
are repeated measurements, not 192 independent test samples.

The primary contrast is the selected candidate minus the strongest supervised
baseline at **15 seconds**:

| Animal | Final dates | Selected R² | Baseline R² | Difference | Negative-transfer days, selected/baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Indy | 5 | 0.34444792 | 0.32411897 | +0.02032895 | 0/0 |
| Loco | 7 | 0.13288026 | 0.15365121 | -0.02077095 | 0/2 |

The equally weighted animal mean difference was **-0.00022099914909815**.
The fixed rule required at least **+0.02** and nonnegative improvement in each
animal, so both gain conditions failed. The negative-transfer-count conditions
were met. This is a failed final outcome, not evidence that the selected method
is stronger overall.

Negative transfer means R² below the **same-day frozen source decoder**. It does
not mean R² below zero, and it is a different comparison from improvement over
the supervised baseline. A post-hoc audit of the saved scores found that the
candidate beat the supervised baseline on **4/5 Indy dates and 2/7 Loco dates**.
The two harmful Loco baseline dates were only 0.0027003 and 0.0049064 R² below
frozen source ridge; avoiding those small deficits did not offset losses on
other dates.

At 15 seconds, the coverage fallback was not triggered on any final date:
all twelve selected scores were strictly above frozen source ridge. Therefore
the lower negative-transfer count cannot be attributed to the coverage guard.
These are descriptions of saved scores, not additional predictions or new
experiments.

Each animal's source decoder was fitted independently on its own labeled
recordings: **two source dates for Indy and three for Loco**. The method and
hyperparameters selected on Indy were then applied using each animal's own
source fit and the permitted target prefix. This tests transfer of calibration
choices; it is not zero-shot transfer of a fitted decoder between animals.

The [freeze](results/frozen.json) preceded final access. The
[download provenance](results/download_provenance.json) records the official
files, pinned sizes, and verified MD5 checksums. A fresh checkout of `72fc834`
reproduced all four comparisons and 192 daily/budget rows to `1e-7`:
[`matches_original: true`](results/reproduced/report.json),
[checkout provenance](results/reproduced/final_checkout.json).
Numerical reproduction does not change `outcome.passed: false`.

## Completed development evidence

The reference remains full neural residual ridge with alpha 10 and update
strength 0.5: mean development R² **0.35010946184569786**, with one day below
the frozen source decoder.

| Phase | Configurations | Best candidate mean R² | Outcome |
| --- | ---: | ---: | --- |
| PCA/affine study | 24 references + 8 candidates | 0.342235 | Gate failed |
| Initial MAP continuation | 8 | 0.343804 | Gate failed |
| SIMPLS continuation | 18 | 0.340665 | Gate failed |
| Source-anchored encoding | 12 | 0.311715 | Gate failed |
| Task-residual continuation | 9 | 0.350887 | Gate failed |
| Shared-channel reweighting | 16 | 0.369517 | Gate failed; 2 harmful days |
| Coverage-guarded reweighting | 1 executed of 2 declared | 0.383216 | Development passed; final failed |

The [initial MAP evidence](../calibnext-map-initial/results/decision.json) is
preserved with its exact source at commit
`b7d301513a5ff15ff71ec4451838164208fcd87d`. All eight attempts numerically
reproduced to `1e-7`. The [SIMPLS/encoding evidence](../calibnext-linear-encoding/results/decision.json)
is preserved at commit `443a0602886496888077907ae3ea5f4acf81f9d1`.
A fresh checkout of that source reproduced **30 configurations and 270 daily
rows**, with [`matches_original: true`](../calibnext-linear-encoding/results/reproduced/development.json).
Historical code and paths must be used for historical replay; archived input
hashes are not hashes of the revised implementation.

These dates have now informed **32 original + 38 archived + 26 current = 96
configurations**. Of these, 72 are counted candidate attempts, including one
baseline-equivalent control in the task-residual batch. The 26 current attempts
are nine task-residual, sixteen shared-channel, and one coverage-guarded
configuration. The second declared coverage configuration was not executed.
Two additional post-hoc diagnostic configurations exposed the same nine days,
producing 18 separately disclosed daily scores. Original pilot and
regularization-range diagnostics are also retained. Method changes and new
batches do not reset development exposure.

The [archived shrinkage diagnostic](../calibnext-linear-encoding/results/shrinkage_bounds.json)
reconstructs each fixed linear correction's R² quadratic from its frozen,
half-strength, and full-strength results. Even the best continuous strength
cannot exceed 0.350865 for the four tested MAP heads or 0.343767 for the nine
tested SIMPLS heads. These are post-hoc analytical development bounds, not newly
scored candidates; they do not bound other priors or algorithms.

The [residual-strength oracle](results/residual_strength_oracle.json) also
reanalyzes saved development scores. It opens no new raw recordings and computes
no new real predictions. Its use of scoring outcomes makes it an exploratory
bound, not a deployable prefix-only selector, an additional scored candidate,
or independent validation.

## Development selection

The [first passing candidate](results/decision.json) is
[`covered_gain`](../../methods/covered_gain.py), with `alpha=0.01`,
`bias_penalty=1`, and `strength=1`. Its mean R² is **0.38321606390802**, an
absolute improvement of **0.03310660206232213** over the fixed baseline
**0.35010946184569786**. Both have **one negative-transfer day out of nine**.
This clears the unchanged development margin and harm rule; it is not a final
generalization result.

The method computes the covariance of labeled prefix velocity after whitening
with the fixed source velocity factor. If its smallest-to-largest eigenvalue
ratio is below `0.01`, or its largest eigenvalue is nonpositive, it keeps the
frozen source decoder. Otherwise it performs shared channel reweighting.
Everything used by this guard comes from the first 15 seconds and fixed source
statistics. The threshold was motivated by earlier failures on these reused
development dates; the observed gain does not establish a causal explanation
or a validated reliability threshold.

The [coverage manifest](batches/003-coverage-guard.json) declared two ordered
configurations. The first passed, so selection stopped before its second
configuration (`bias_penalty=0`). No later batch is authorized by that result.

## Earlier current-version hypotheses

The declared [001-task-prior batch](batches/001-task-prior.json) uses
[`task_residual`](../../methods/task_residual.py): a joint residual fit on the
full neural features and the two frozen source-prediction axes. Neural residual
coefficients retain penalty 10; source-prediction coefficients have a separate
penalty. Both blocks are standardized using only the observed prefix. Because
the prediction block is a linear function of the same neural inputs, it changes
the shrinkage geometry without adding target information.

The nine ordered configurations began with a limiting control:
`alpha=1e16`, `bias_penalty=0`, `strength=0.5`, equivalent to the inherited
baseline. Then `alpha={1,10,0.1,100}` was tested in that order, each at
`strength={0.5,1}`. Best mean R² was **0.35088667231396226**; none passed.

[`shared_gain`](../../methods/shared_gain.py) implements shared channel
reweighting across causal lags and both output axes. Its coefficients may be
negative, so they are not restricted to physical firing-rate gains. Its
[sixteen-configuration batch](batches/002-shared-gain.json) reached mean R²
**0.3695165217122956**, a gain of **0.0194070599**, with **two**
negative-transfer days. It missed both the required margin and the harm rule.
These statistical mechanisms are not presented as novel algorithms.

[`temporal_residual`](../../methods/temporal_residual.py) and
[`blocked_residual`](../../methods/blocked_residual.py) are implemented but
**unscored**. Their presence in the repository is not evidence of performance
and does not add scored configurations to the counts above.

New algorithms may receive only the fixed source preparation and the current
labeled prefix. Each trial binds the scientific core plus its own method file.
Adding an unused method does not invalidate earlier trials; changing an already
used method does. Freeze and replay verify every used method, including failed
and crashed attempts. Methods cannot supply a different scorer or access final
labels.

## Fixed gate and interpretation

The development gate remains mean candidate R² **at least
0.37010946184569786**, with at most **one** negative-transfer development day.
All nine dates use the same macro x/y R² definition and scoring begins at
30.5 seconds. The prefix is the first 15 elapsed seconds, not a movement-selected
subset. Features remain 20 ms bins, 10 causal lags, 60 ms causal EMA, and source
ridge alpha 10.

The [diagnostic evidence](results/diagnosis.json) shows strongly correlated
prefix samples and a neural participation rank of 3.78–21.04 despite 741 valid
bins. Participation rank is not an inferential sample size. Mean-only and
slope-only corrections both underperformed the full residual reference. These
observations motivate further tests without establishing a causal explanation
for failure.

The prespecified final rule required, at 15 seconds, nonnegative mean improvement
in each animal, at least
0.02 improvement when the two animals are averaged equally, and no increase in
negative-transfer days within either animal. Final comparisons retain the
development-selected baseline and use separate source models for the animals.
The failed final result is retained and cannot trigger retuning.

## Run and reproduce

From the repository root, Python 3.12 and the existing locked environment are
required. The published study is frozen and already has a final report. The
CLI therefore automatically reproduces its **four frozen final comparisons**:

```sh
uv sync --locked
uv run --no-sync python -m calib15 download
uv run --no-sync python -m calibnext reproduce
```

The development download contains only Indy source and development files.
Final replay also needs **14,052,108,936 bytes (14.05 GB)** for twelve final
recordings and three Loco source recordings, plus the existing development data
and working space. Raw data are excluded from Git. The replay verifies the
frozen access marker and input hashes before downloading or loading these files.
Results are written to `studies/calibnext/results/reproduced/report.json`.
The published fresh-checkout replay matched all 192 rows to `1e-7`.

The current streaming evaluator reproduced the inherited development baseline
at mean R² `0.3501094618457011`, with maximum daily error `1.64e-14`.
All **26 current development configurations and 234 daily rows** reproduced
from a fresh checkout of `60122c5` to `1e-7`:
[`matches_original: true`](results/reproduced/development.json),
[checkout provenance](results/reproduced/checkout.json). That run used its own
checked-out code and the locked environment.

To repeat those development results after the final report exists, call the
development-only API explicitly, pinning compute threads before numerical
libraries are imported:

```sh
uv run --no-sync python -c "import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
from pathlib import Path
from calibnext.replay import reproduce_development
print(reproduce_development(Path.cwd()))"
```

This writes `studies/calibnext/results/reproduced/development.json` without
adding search events. Scientific fields must agree within `1e-7`; elapsed times
may differ. `matches_original: true` in replay evidence reports numerical
agreement; `outcome.passed: false` in the final report records failure of the
scientific success criterion.
Reproducing the failure does not turn it into a successful method.

Do not call `sanity`, `batch`, or `freeze` to rerun this published experiment.
They are lifecycle operations for an unfinished study; the saved selection is
immutable. No further candidate may be selected or tuned using these final data.

To reproduce either historical continuation, use a separate checkout of its
source commit listed above. At those commits, the logs occupy their original
`studies/calibnext` path. Run the same locked setup, development download, and
`python -m calibnext reproduce` there. At those historical commits it dispatches
to development replay. Current source rejects using those ledgers as new search.

The freeze was committed and pushed at `60122c5` before the final access marker.
Its 64 scientific input hashes match the committed bytes. Local software
verification passed **142 tests**, lint, formatting, and type checks;
[Ubuntu/Windows CI passed](https://github.com/Nigmat-future/neurocalib-bench/actions/runs/39160460675).
These checks and replay results verify software and reproducibility, not the
neuroscience hypothesis. Read the [protocol](PROTOCOL.md) and
[research loop](program.md) for the complete contract.
