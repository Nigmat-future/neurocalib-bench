# Continued fifteen-second calibration study

This is a continuation after the failed `calib15-v2` search, not an independent
replication or a fresh development set. The same nine Indy development dates
have already informed 24 baselines and eight PCA/affine candidates. The earlier
range diagnostic and original six-day pilot are also retained. Continued search
increases development selection bias; only the still sealed final recordings can
test generalization.

An additional eight MAP configurations failed in the first continuation. Their
unaltered evidence is archived in `studies/calibnext-map-initial`; commit
`b7d3015` preserves the exact original paths and source for replay. Two post-hoc
diagnostic configurations also exposed the same nine development dates. This
revision fixes evidence-overwrite and replay-integrity guards before continuing
with PLS and an additional source-anchored encoding mechanism. It does not reset
or conceal the prior eight attempts, change the baseline, or relax the gate.

## Unchanged contract

The original `studies/calib15/protocol.json` determines every recording, checksum,
animal, and access role. The source-only choice in its `results/features.json`
is fixed: 20 ms bins, 10 causal lags, 60 ms causal EMA, source ridge alpha 10.
Features never cross a recording gap. Calibration uses only the first 15 elapsed
seconds, not 15 seconds of selected movement. All development scoring starts at
30.5 seconds. No extra target labels, future neural samples, or development-day
training enter an individual day's calibration.

The reference is the already selected strongest same-label baseline, full neural
residual ridge alpha 10 with update strength 0.5. Its nine-day macro R² is
0.35010946184569786 and it harms one day relative to the frozen source decoder.
The gate remains mean candidate R² minus this baseline at least 0.02, with no
increase in negative-transfer days. No dates, axes, or difficult prefixes may be
dropped. No alternative aggregation or rescaled gain may replace this gate.

## Methods and ordered research batches

Three mechanisms are implemented before this revision's candidate scoring:

1. Source-only SIMPLS regression directions identify a neural
   subspace using source behavior as well as neural covariance. Prefix residual
   ridge is then fit in that fixed subspace. This differs from unsupervised PCA
   and from the rank-at-most-two space of source output affine adaptation. Neural
   inputs use source standardization; source velocity is centered without axis
   standardization. An orthonormal basis preserves the learned SIMPLS span.
2. A source-anchored MAP residual uses the fixed source center and scale. Its
   objective is prefix mean squared residual plus alpha times a weight penalty,
   plus a separate penalty on the intercept correction. The weight precision is
   either identity or `(source_covariance + 0.1 I) / 1.1`. This matrix, not its
   inverse, penalizes changes to directions identified by source data. These are
   established statistical mechanisms; this study does not claim a new algorithm.
3. Source-anchored neural encoding updates the conditional mapping from whitened
   velocity to source-standardized neural features with a two-slope and intercept
   ridge fit. Source neural noise and velocity priors stay fixed. The source
   joint covariance is constructed so its initial LMMSE decoder equals the fixed
   source ridge exactly. Calibration changes the encoding coefficients and
   intercept; the output correction is shrunk toward that exact source readout.
   Calibration never estimates noise or velocity priors from scoring labels.

Each finite batch is declared in an ordered manifest before its first trial.
Every trial records its hypothesis, configuration, code and protocol hashes,
manifest hash, environment, start, terminal state, and all nine daily results.
Failures and interrupted configurations are retained and cannot be rerun as new
successes. A failed batch may be followed by another declared batch; prior
exposures remain in the cumulative ledger. The first passing candidate ends the
search immediately, even if later declared configurations might score higher.

## Final access and success

Before a passing gate, no final recording or Loco recording may be downloaded or
opened. A passing development decision seals the exact code, inputs, winner,
source features, ledger, and inherited comparisons before final access starts.
Then fit separate source models for Indy and Loco; matching channel names across
animals never identify the same electrode. The five Indy and seven Loco final
recordings receive the frozen method and fixed reference comparisons at 0, 5,
15, and 30 seconds. The primary result remains 15 seconds.

Final success requires nonnegative gain in each animal, no increase in negative
transfer in either animal, and an equally weighted two-animal mean gain of at
least 0.02. A final failure is reported as a failure. No final labels may guide a
new method, stopping decision, or hyperparameter change. Numerical reproduction
uses the same frozen comparisons and a tolerance of 1e-7, excluding elapsed time.
