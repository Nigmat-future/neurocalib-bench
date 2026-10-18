# neurocalib-bench

**Cross-day neural decoding under fixed calibration budgets.**

A reproducible benchmark of cross-day intracortical velocity decoding, with an original pilot and a frozen supervised calibration study. Compare frozen ridge regression, distribution alignment, and supervised readout correction under **0, 5, 15, and 30 seconds** of new-day data.

The experiments use public recordings and offline decoding. The original pilot reports two final-test days from one animal; the later study reports twelve different final dates across two animals. The contribution is controlled experiments and inspectable evidence, with development separated from final testing. No new state of the art, clinical validation, or Neuralink replication is claimed.

中文概述：固定新日可见数据与校准时间，比较简单适应方法；先冻结协议与配置，再评估保留测试日。保留无收益和负迁移结果。

## Fifteen-second supervised calibration study

**Final outcome: failed.** At the primary 15-second budget, the frozen candidate
improved mean R² by **+0.020329 on Indy** but reduced it by **-0.020771 on Loco**.
The equally weighted animal mean difference was **-0.000221**, below the required
**+0.02**. Loco also failed the required nonnegative per-animal gain. Inspect the
[complete final report](studies/calibnext/results/final/report.json) and
[study interpretation](studies/calibnext/README.md#frozen-final-outcome-failed).
The candidate is retained; selection and tuning have ended.

The [second study](studies/calib15/README.md) compares 24 tuned supervised baselines
with agent-proposed calibration models under the same 15-second labeled prefix.
It reserved nine new development dates and twelve final dates across two animals.
A +0.02 development R² gain and no additional negative-transfer days are required
before final access. The study documents a pre-candidate regularization-range
amendment and preserves its [initial range diagnostic](studies/calib15-rangecheck/README.md).

The completed run tried eight candidates. Best supervised baseline R² was
**0.350109**, versus **0.342235** for the best candidate: **ΔR² = -0.007874**.
This phase failed the development gate and did not freeze a candidate or access
the final recordings. [Inspect its complete evidence](studies/calib15/README.md).
All 32 configurations reproduced from a fresh Git checkout to `1e-7`; 61 tests and
[Ubuntu/Windows CI](https://github.com/Nigmat-future/neurocalib-bench/actions/runs/39146578808)
passed.

The [continued study](studies/calibnext/README.md) retained the same reference,
gate, nine development dates, and previously sealed final dates. Eight MAP attempts reached
at most **0.343804** mean development R². A subsequent 18 SIMPLS attempts reached
**0.340665**, and 12 source-anchored encoding attempts reached **0.311715**.
All failed the development gate. The [eight-MAP archive](studies/calibnext-map-initial/README.md)
and [30-attempt SIMPLS/encoding archive](studies/calibnext-linear-encoding/README.md)
preserve their original evidence and source commits. The latter reproduced all
30 configurations and 270 daily rows from a fresh checkout:
[`matches_original: true`](studies/calibnext-linear-encoding/results/reproduced/development.json).

The current version supports separately hashed, single-file candidate methods.
Nine task-residual configurations reached **0.350887** mean development R²;
16 shared-channel configurations reached **0.369517**, with two negative-transfer
days. Both batches failed the fixed gate. The first
[`covered_gain` candidate](studies/calibnext/results/decision.json) then passed:
**R² 0.383216**, **ΔR² +0.033107**, and **one of nine** days below frozen ridge,
matching the reference's harm count. Its prefix-only coverage guard retains the
source decoder when labeled motion has insufficient directional coverage.

Selection stopped immediately. The second declared coverage candidate and the
implemented temporal/blocked residual methods remain unscored. The same dates
have now informed **32 original + 38 archived + 26 current = 96 configurations**,
including one baseline-equivalent control. This is 72 counted candidate
attempts, including that control, plus **two separately disclosed post-hoc
diagnostic configurations**. Strength-oracle calculations only reanalyze saved
results. This repeated development selection was not independent validation:
the **+0.033107 development gain did not meet the later two-animal final rule**.
The final comparison contains four fixed methods, twelve dates, and four budgets
(192 rows). Negative transfer means doing worse than the same-day frozen source
decoder, not having R² below zero. Lower negative-transfer frequency did not make
the candidate stronger overall.

Each animal had its own source decoder fitted on its own labeled recordings:
two source dates for Indy and three for Loco. What transferred across animals was
the selected method and its hyperparameters, not a zero-shot fitted decoder.
All 26 current development configurations and all four frozen final comparisons
reproduced from fresh checkouts to `1e-7`: [final numerical replay](studies/calibnext/results/reproduced/report.json),
[checkout provenance](studies/calibnext/results/reproduced/final_checkout.json).
`matches_original: true` means numerical agreement, while `outcome.passed: false`
records failure of the scientific criterion. These are separate claims.

The original pilot's locked results follow below.

## Locked pilot results

Eight development candidates were recorded before freezing the final comparison. The selected configuration was strongly shrunk CORAL: `alpha=1`, `lags=10`, `strength=0.25`, `covariance_shrinkage=0.9`. Its development objective was **0.396156956**, compared with **0.383974819** for default frozen ridge.

The table below averages macro x/y velocity R² equally over the **two final-test days and budgets 5, 15, and 30 seconds**, as declared before testing. Zero-budget results remain in the full reports.

| Frozen comparison | Mean test R² |
| --- | ---: |
| Selected shrunk CORAL (`winner`) | 0.095010 |
| Matched frozen / default frozen ridge | 0.064023 |
| Default center | 0.131460 |
| Default standardize | 0.132234 |
| Default full-strength CORAL | -0.648762 |
| Default supervised correction | 0.323263 |

The primary adaptation effect, selected CORAL minus its matched frozen reference, was **+0.030986568 absolute R²**. The development winner did **not** rank first on the test days: cheaper centering and standardization performed better. The winner was retained as selected; it was not replaced after seeing test scores. Supervised correction used velocity labels and is a different information setting.

At a 30-second prefix, the selected unsupervised pipeline still failed on the October test day, with **R² below zero**:

| Test recording | Frozen R² | Selected R² | Supervised R² |
| --- | ---: | ---: | ---: |
| `indy_20161013_03` | -0.150143 | -0.112736 | 0.399160 |
| `indy_20161212_02` | 0.278190 | 0.298256 | 0.449756 |

These are two recording days from one animal. They support a descriptive pilot result, with a clear failure case, rather than a state-of-the-art or population-level claim.

![Locked test results: the selected pipeline improves on matched frozen ridge on average, but centering and standardization do better; its October test-day R² remains negative at every prefix budget.](results/final/calibration_curves.png)

Inspect the [full final JSON](results/final/report.json), [final CSV](results/final/report.csv), [frozen configuration and hashes](results/frozen.json), [eight-candidate summary](results/search_summary.csv), and [complete trial ledger](results/experiments.jsonl). Raw recordings are excluded from the repository.

## Reproduce the original pilot

Read the [protocol](PROTOCOL.md) and inspect the [recording manifest](protocol.json). Source training uses two recordings; two more support development; two later recordings are reserved for final evaluation. Every method is scored from 30.5 seconds onward on identical bins.

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. Core dependencies are NumPy, h5py, Matplotlib, and Pydantic. No GPU or language-model API is required.

From the repository root:

```sh
uv sync --locked
uv run --no-sync python -m neurocalib download
uv run --no-sync python -m neurocalib reproduce
uv run --no-sync python -m neurocalib report
```

`download` retrieves and verifies the six pinned files. `reproduce` recomputes only the saved final configurations and writes [results/reproduced/report.json](results/reproduced/report.json), leaving the original final results intact. Scientific values must match within relative and absolute tolerance `1e-7`; machine-dependent timings are excluded from that comparison. `report` renders the saved original final scores. On a new machine, use `reproduce`: `final` is reserved for the original run and checks its recorded runtime.

Use `uv run --no-sync python -m neurocalib --help` for the complete command list.

Development checks use the same locked environment:

```sh
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync basedpyright neurocalib calib15 calibnext methods
uv run --no-sync python -m unittest discover -s tests -v
```

The current continuation implementation passed **142 unittest tests**, lint,
formatting, and type checks before final access.
[Ubuntu/Windows CI passed](https://github.com/Nigmat-future/neurocalib-bench/actions/runs/39160460675).
[Ubuntu and Windows CI also passed
on the initial pilot implementation](https://github.com/Nigmat-future/neurocalib-bench/actions/runs/39142899064).
Tests use synthetic HDF5 fixtures and do not download the research recordings.
Passing those tests establishes software checks, not evidence for the neuroscience
hypothesis. For the frozen calibration study and its historical source checkouts,
use the [continuation reproduction instructions](studies/calibnext/README.md#run-and-reproduce).

The original pilot's numerical reproduction passed for all seven frozen comparisons: [`matches: true`](results/reproduced/report.json), with relative and absolute tolerance `1e-7`. A fresh Git checkout also reproduced all seven comparisons, verifying that the committed files preserve the frozen evidence hashes.

## Original pilot comparisons

| Method | Information from the new-day prefix |
| --- | --- |
| Zero prediction | None |
| Frozen ridge | None |
| Center | Spike means |
| Standardize | Spike means and variances |
| Regularized CORAL | Spike means and covariance |
| Supervised readout correction | Spikes and velocity labels |

All budgets measure elapsed recording time. Supervised correction receives labels unavailable to the unsupervised methods; equal seconds do not mean equal information. No prefix is selected using movement speed, target identity, or success.

Inputs are M1 channel spike counts in 20 ms bins with at most 400 ms of causal history. Target velocity is in mm/s. A source-only channel-name set is reused across days, with missing channels reported and zero-filled. Prediction follows prefix fitting; this is not a closed-loop cursor experiment.

## Original pilot sanity control

The default frozen decoder uses ridge `alpha=1` and 10 bins of history. Its source-label control circularly shifts the concatenated source velocity labels by half their length before fitting; development labels stay unchanged.

| Development recording | Frozen R² | Shifted-source R² |
| --- | ---: | ---: |
| `indy_20161006_02` | 0.395757204 | -0.002746809 |
| `indy_20161007_02` | 0.372192435 | -0.016684721 |

These are macro x/y R² on the same scoring intervals, not squared correlations. Frozen predictions are identical across calibration budgets. Inspect the [baseline values](results/baseline.json), [baseline CSV](results/baseline.csv), and [source-shift control](results/source_shift_control.json). The deterministic shift is a diagnostic check, not a permutation significance test or final-test evidence.

## AI-assisted research

An external coding agent can follow [program.md](program.md), propose a configuration and hypothesis, run development-only trials, and inspect recorded outcomes. A final freeze binds the selected configuration, scientific code, protocol, and environment before test scores are opened. A fixed parameter sweep is described as a sweep, not autonomous intelligence.

The lifecycle commands are:

```sh
uv run --no-sync python -m neurocalib baseline
uv run --no-sync python -m neurocalib trial --config configs/02_center.json --hypothesis "Prefix mean matching may reduce an additive shift; it may fail when early movement differs from later movement."
uv run --no-sync python -m neurocalib freeze
uv run --no-sync python -m neurocalib final
```

This lifecycle is for an **unfrozen research run**, not a command sequence to rerun on the published final experiment. The published checkout is already frozen and refuses further trials. The original pilot's search allowed at most 12 distinct candidates and selected mean development R² over both days at 5, 15, and 30 seconds. Supervised correction was a fixed reference, excluded from search. All eight attempted pilot candidates, including the failed full-strength CORAL hypothesis, remain in the [experiment ledger](results/experiments.jsonl).

For a new study, create a separate working copy, preserve the original records, and predeclare genuinely untouched test dates. Deleting a freeze and continuing to tune against the same opened test recordings does not create a valid new holdout.

The executable experiment does not host or call an LLM. Raw recordings stay out of Git; derived metrics and provenance form the reviewable output.

## Data and attribution

The original pilot's six selected files total **635,651,025 bytes**. Downloads must match the sizes and MD5 digests in `protocol.json`; the supervised study has its own [recording manifest](studies/calib15/protocol.json).

The continuation's final replay needs **14,052,108,936 bytes (14.05 GB)** of
additional final/source recordings, plus the existing development data and
working space. See its [verified download provenance](studies/calibnext/results/download_provenance.json)
and [reproduction commands](studies/calibnext/README.md#run-and-reproduce).

O'Doherty, J. E., Cardoso, M. M. B., Makin, J. G., and Sabes, P. N. *Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology*, version 2 (2020). [Dataset and license](https://zenodo.org/records/3854034), DOI [10.5281/zenodo.3854034](https://doi.org/10.5281/zenodo.3854034).

The data and published derived scientific results are **CC BY 4.0**. This repository's software is **MIT**. Attribute the original dataset and the analysis when reusing derived metrics or plots. Cite the original dataset separately from the software; see [CITATION.cff](CITATION.cff).

## Research context

[FALCON](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html) already studies consistent few-shot neural decoding. [NoMAD](https://www.nature.com/articles/s41467-025-59652-y) and [PRI-T](https://www.nature.com/articles/s41551-025-01536-z) provide substantially broader evidence for unsupervised stabilization and recalibration. This repository asks a narrower budget-controlled question; it does not establish superiority over those methods.

The pilot's published final results cover only two days from one animal; the
later study reports twelve final dates across two animals and failed its fixed
success criterion. Report
day-level results, negative transfer, measured compute costs, and failed
hypotheses. Do not infer population-level significance from autocorrelated 20 ms
bins or repeatedly reused development dates.
