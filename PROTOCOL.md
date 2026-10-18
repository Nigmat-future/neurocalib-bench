# Experimental protocol

Protocol identifier: `indy-pilot-v1`. Written before final-test evaluation.

This is a locally frozen pilot protocol, not a registered preregistration. The machine-readable recording manifest is [protocol.json](protocol.json). Configuration files define the permitted model parameters. The final freeze must bind the protocol, configuration, implementation, and environment hashes before any final-test scores are computed.

## Question and scope

How much does simple cross-day adaptation change two-dimensional cursor-velocity decoding when each new recording exposes only a fixed initial calibration interval? We compare zero-label moment matching with supervised linear readout correction under the same time budgets.

The primary comparison is the paired change in velocity R² between the development-selected winner and `matched_frozen`: exactly the winner's configuration with only `mode` changed to `frozen`. This separates adaptation from changes caused by selecting ridge strength or history. Improvement is a hypothesis, not an expected result. Negative transfer is an outcome to preserve.

This study uses offline recordings of natural reaching from one nonhuman primate. It does not measure closed-loop BCI control, human motor intention, clinical utility, or replication of Neuralink. It introduces neither a new public benchmark dataset nor the first causal neural decoder. Novelty beyond this restricted empirical comparison is unestablished.

## Data fixed in advance

The data source is O'Doherty, Cardoso, Makin, and Sabes, *Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology*, [Zenodo version 2](https://zenodo.org/records/3854034), DOI [10.5281/zenodo.3854034](https://doi.org/10.5281/zenodo.3854034). The upstream dataset contains spike timestamps and cursor/target position, with session-level channel identifiers. Cursor coordinates are in millimetres and behavioral timestamps in seconds. Data are licensed CC BY 4.0; software licensing does not replace the data license.

The pilot uses these six recordings, selected before comparing decoder performance:

| Role | Recording |
| --- | --- |
| Source training | `indy_20160930_05.mat` |
| Source training | `indy_20161005_06.mat` |
| Development | `indy_20161006_02.mat` |
| Development | `indy_20161007_02.mat` |
| Final test | `indy_20161013_03.mat` |
| Final test | `indy_20161212_02.mat` |

Their declared combined download size is **635,651,025 bytes**. Exact file sizes and upstream MD5 digests are recorded in `protocol.json`; verify both before loading. Downloading a test file for integrity verification does not authorize inspecting its behavior or computing its scores before freezing.

Do not substitute dates, remove a low-performing date, or select channels using development or final-test behavior. Any structural data failure must be recorded. Changing the manifest creates a new protocol rather than silently replacing this experiment.

## Inputs and causal timing

- Use M1 channels only. Select the common channel-name set from the two source recordings, in a deterministic order. Sum the available unit spike trains within each selected channel; do not assume that a sorted unit has a stable identity across days.
- Align a target recording by these source channel names. A missing source channel is zero-filled and the missing-channel count is reported. Extra target channels are ignored. This is input alignment, not a claim that recorded neurons persist across days.
- Bin spikes into nonoverlapping **20 ms** intervals. A prediction emitted at a bin end may use that bin and earlier bins only.
- Construct a causal history of at most **20 bins (400 ms)**; the default is **10 bins (200 ms)**. History must never join the end of one source recording to the beginning of another.
- Compute x/y cursor velocity from backward differences of cursor position at bin boundaries, in **mm/s**. No centered derivative, future neural bin, bidirectional smoother, or future behavioral state is an input. Ground-truth velocity is used for scoring and for the explicitly supervised comparator only.
- Fixed prefix budgets are **0, 5, 15, and 30 seconds**, measured from the start of the usable recording. A budget selects an elapsed-time prefix, not an equal count of movement samples. Do not filter that prefix by target, speed, reach success, or behavior.
- The shared scoring region starts at **30.5 seconds** for every method and every budget, and ends at the last usable bin of that recording. The 0.5-second gap exceeds the maximum history. No method gets an easier scoring segment because it used less calibration data.

“Causal” describes prediction features after the prefix is available. Adaptation is an offline fit on the permitted prefix; this is not continuous online learning. Prefix adaptation time and prediction time are different measurements and must be reported separately.

## Comparators

All learned methods start from the same source-trained ridge regression family and the same feature-generation code. The five default learned references use `alpha=1`, `lags=10`, `strength=1`, and `covariance_shrinkage=0.1`. Ridge minimizes mean squared training error plus `alpha` times squared readout norm in source-standardized coordinates. Ridge strength, history, adaptation strength, and covariance regularization are configuration values selected on development recordings only.

1. **Zero prediction:** predicts zero velocity on the common scoring region. It is a reference, not the R² denominator.
2. **Frozen ridge:** no target adaptation. Source centering and source fit parameters remain fixed.
3. **Center:** estimate target-prefix means and translate target features toward source means.
4. **Standardize:** match per-feature prefix means and variances to source statistics, with numerical protection for low-variance features.
5. **Regularized CORAL:** match prefix covariance to source covariance through regularized whitening and recoloring. Regularization is essential when a short prefix has fewer effective observations than feature dimensions. Fit this transform only on the permitted neural prefix.
6. **Supervised readout correction:** use target-prefix neural data **and its velocity labels** to correct the source linear readout. This comparator uses the same elapsed-time budget but strictly more information than the unsupervised methods. Its results must be labelled supervised, not calibration-free.

At zero budget, target-adaptive methods fall back to the frozen source model. The final comparison contains exactly seven labelled configurations: the five default learned references, the selected `winner`, and `matched_frozen`. Retain all seven labels even if two configurations are numerically identical. Zero-prediction scores are included as an ancillary reference for every evaluation cell, not as an eighth fitted method.

The development `source_shift_control` is deterministic: concatenate the valid source-training velocity labels in the declared source-file order and circularly shift that array by `floor(n_source_samples / 2)` rows, leaving the source neural features fixed. Fit the otherwise default frozen decoder on those shifted labels. Development labels are unchanged. This disrupts local neural/behavioral alignment while retaining temporal structure; it is not an independent-label permutation or a formal significance test. Its purpose is diagnosis, not candidate selection, and no stochastic seed is involved.

Whole-session unsupervised alignment is a valid alternative research setting, but it is excluded here. Its access to more new-day data should not be described as leakage unless it violates its own declared protocol.

## Development and AI-assisted search

Source recordings fit decoder parameters. Development recordings support candidate selection and error diagnosis. Their labels may be used to score candidates, but only the supervised comparator may use prefix labels for adaptation. Missing or non-finite behavioral labels may exclude rows from supervised fitting and scoring; they must not select the neural observations available to an unsupervised prefix fit. Preserve time gaps and report the actual number of prefix and scoring bins.

The AI-assisted loop in [program.md](program.md) proposes a hypothesis and configuration, runs a development-only trial, reads the recorded outcome, and retains or rejects that candidate. The ledger must retain unsuccessful trials. A grid search is a fixed sweep, not an autonomous research claim. No self-hosted language model or API key is needed to run the benchmark itself.

The selection objective is the arithmetic mean of macro x/y R² over the **two development recordings and budgets 5, 15, and 30 seconds**, giving six equally weighted cells. The zero-second condition is reported but excluded from selection. Do not select using runtime, final-test scores, RMSE, or a different aggregate.

Search is restricted to `frozen`, `center`, `standardize`, and `coral` configurations accepted by the validated schema. The supervised comparator is a fixed final reference and is excluded from search. Attempt at most **12 distinct canonical configurations**; crashes consume that budget, and duplicate configurations reuse their completed result. The recorded limit cannot be increased during the run. Stop earlier when development evidence gives no reason to continue. Do not let an agent change date splits, budgets, scoring interval, channel selection, or metrics to improve its score. Repeated development use can still overfit these two days; report that limitation and the number of tried candidates.

## Freeze and final-test access

Before the first final-test run:

1. Complete development and choose the candidate configuration.
2. Save a freeze artifact containing content hashes for the selected configuration, `protocol.json`, this document, scientific implementation, and the locked environment, together with the candidate-selection evidence.
3. Record the winner, all five default learned references, and the matched frozen reference. These methods are deterministic closed-form computations, with no seed search. Freeze validation must fail after a covered input changes.
4. Run the final evaluation without further configuration selection. Preserve its complete output and provenance, including poor results and missing-channel notices.

Neither final-test labels nor final scores may guide another candidate. A repair after final evaluation must be disclosed with the reason, changed files, previous output, and a new freeze; the repaired evaluation is no longer a pristine first look. This is a reproducibility boundary in a local workflow, not access-control protection against a person deliberately opening a file.

## Metrics and interpretation

Report x and y velocity R², their macro-average as the primary accuracy metric, and overall velocity RMSE in mm/s on the common scoring bins. R² uses each axis's held-out mean as its denominator reference; it may be negative. Do not clip negative values or replace R² with squared Pearson correlation. A constant ground-truth axis or non-finite prediction must raise an error rather than silently produce a score.

For each method and budget, retain recording-level results. The primary adaptation delta is `winner.r2 - matched_frozen.r2` for the same recording and budget. Comparisons with the default frozen ridge are secondary because they also include hyperparameter selection. Report the number of negative-transfer days. With only two final-test days from one animal, descriptive comparisons are the appropriate primary analysis; do not treat thousands of autocorrelated bins as independent replicates, or claim population-level significance from them.

Record adaptation wall time and causal prediction cost, including CPU/platform, thread limits, number of features, and timing boundaries. Measure prediction cost as the mean wall time of **200 actual single-bin `predict` calls** on the first scored bins, or all scored bins if fewer than 200 are available. Each history vector is already constructed before the timed calls; this measurement excludes history construction, buffering, source fitting, and data I/O. Prefix calibration wall time is timed separately. Batch prediction throughput is not an interactive per-bin latency measurement. These are measured costs on the reported machine, not implant power or an enforced real-time guarantee. Include buffering/history duration when discussing response time.

Raw recordings stay outside Git. Publish derived metrics, plots, frozen configurations, trial records, integrity checks, and environment/code provenance. A plot without its underlying values and scoring definition is insufficient evidence.

## Closest prior work and limits

- [FALCON (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html) already evaluates consistent few-shot neural decoding. This pilot is a smaller, explicitly budgeted comparison, not a replacement.
- [NoMAD (Nature Communications 2025)](https://www.nature.com/articles/s41467-025-59652-y) demonstrates unsupervised alignment using latent dynamics. This pilot does not establish superiority over NoMAD or other deep alignment methods.
- [PRI-T (published online 2025)](https://www.nature.com/articles/s41551-025-01536-z) evaluates unsupervised recalibration with offline, simulated, and online evidence. Offline replay here cannot establish equivalent closed-loop benefits.

The six-day pilot, one animal, fixed source-only channel set, and short prefixes limit generalization. Synthetic fixtures verify software behavior; they do not validate the biological hypothesis. A null or harmful adaptation result is publishable repository evidence, not a reason to change this protocol after seeing the tests.
