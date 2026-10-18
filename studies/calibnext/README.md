# Continued calibration research

This continues the failed [PCA/affine study](../calib15/README.md), using its same
nine Indy development dates, source features, strongest supervised baseline, and
15-second label budget. The twelve final dates remain sealed until a candidate
passes the unchanged development gate. This is repeated development selection,
not a new independent study.

The initial eight MAP trials completed without passing. Best mean R² was
0.34380420268891343 (source precision, alpha 10, full strength, intercept penalty
1), a difference of -0.006305259156784426 from the reference, with one harmful
day. All eight trials numerically reproduced to 1e-7. This initial implementation
is preserved before fixing a CLI issue: calling `sanity` again after trials can
overwrite bound evidence. No final recording was accessed.

## Why the earlier methods failed

The [reproducible diagnostic](results/diagnosis.json) supports several mechanisms,
without establishing a single causal explanation:

- PCA32 retained 97.23% of source prediction energy. Loss of the source mapping
  alone does not explain failure. It represented about 77.59% of the full neural
  residual correction in the source-covariance metric; affine represented 15.59%.
  Some omitted correction may itself be noise.
- Each prefix contains 741 valid bins but has neural participation rank only
  3.78–21.04 and adjacent-bin correlation 0.813–0.869. These bins are not
  independent training examples. Participation rank is not an inferential sample
  size estimate.
- Intercept-only correction gives mean R² 0.328949, slope-only 0.296292, and the
  full residual baseline 0.350109. Both parts matter. On the only negative-transfer
  date, the prefix and scoring residual means point in opposite directions, and
  prefix velocity is almost rank one. One such date cannot establish causality.

Two post-hoc diagnostic configurations were additionally evaluated on all nine
days. Their 18 daily scores are disclosed separately from the previous 32 search
configurations. The [diagnostic script](diagnose.py) has an explicit eleven-file
whitelist and cannot use the final or Loco recordings.

## What is being tested

Source-supervised **SIMPLS** directions replace neural-variance PCA. A second
family performs **source-anchored MAP residual updates** with fixed source
coordinates, paired identity/source covariance precision, and an independently
regularized intercept. Neither family is claimed as a new statistical algorithm.
The source precision is `(C_source + 0.1 I) / 1.1`, not its inverse.

PLS has existing neural-decoding precedents, including
[FENet](https://authors.library.caltech.edu/records/emwed-ap219).
[Kernel transfer](https://www.nature.com/articles/s41467-023-41215-8) already studies
source-plus-residual mechanisms. Recent
[APST](https://arxiv.org/abs/2609.39080) uses labeled association profiles with a
transformer, but its complete-trial budget and variance-weighted R² differ from
this continuous-prefix, macro-axis benchmark. Its scores are not directly comparable.

The fixed gate is candidate mean R² **at least 0.37010946184569786**, with at most
**one** negative-transfer development day. Candidates are declared in finite,
ordered [batch manifests](batches). The first passing candidate ends selection.
All previous attempts remain in the cumulative ledger.

## Run and reproduce

From the repository root, with the existing locked environment:

```sh
uv sync --locked
uv run --no-sync python -m calib15 download
uv run --no-sync python -m calibnext sanity
uv run --no-sync python -m calibnext batch 001-map-prior
uv run --no-sync python -m calibnext gate
uv run --no-sync python -m calibnext reproduce
```

`sanity` reproduced the previous baseline with maximum daily R² error
`1.64e-14`. Causal chunked features are elementwise identical to the original
builder. Source sufficient statistics avoid keeping whole lag-expanded source
recordings in memory. New raw recordings and dependencies are not needed for
development.

`reproduce` repeats all completed continued-development trials without adding
search events. After a completed final, it instead reproduces the frozen final
comparisons. Scientific row fields must agree to `1e-7`; elapsed times may differ.
Do not rerun `sanity` after trials start: its byte-level evidence is bound.

Only after the passing gate:

```sh
uv run --no-sync python -m calibnext freeze
uv run --no-sync python -m calibnext final
uv run --no-sync python -m calibnext reproduce
```

`final` verifies the seal before downloading any sealed recording or Loco source.
It evaluates separate source models for the animals and preserves the original
four-budget comparison. A final failure is retained and cannot trigger retuning.
Read the [protocol](PROTOCOL.md) and [research loop](program.md) for the full contract.
