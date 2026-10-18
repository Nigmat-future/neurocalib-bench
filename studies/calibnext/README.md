# Continued fifteen-second calibration research

This continues the failed [PCA/affine study](../calib15/README.md) on the same nine
Indy development dates, fixed source features, strongest supervised baseline,
and 15-second labeled prefix. **A candidate has now passed the development gate;
two-animal final validation is pending.** At selection, none of the twelve final
recordings had been accessed. The original six-day pilot is a separate,
already completed experiment with its own published final results.

The current version permits new calibration algorithms in separately hashed
single-file methods while keeping evaluation, labels, scoring, and the gate
fixed. This is repeated development selection, not independent validation.

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
| Coverage-guarded reweighting | 1 executed of 2 declared | 0.383216 | Development gate passed |

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

Passing development only authorizes a frozen final comparison. At 15 seconds,
the candidate must have nonnegative mean improvement in each animal, at least
0.02 improvement when the two animals are averaged equally, and no increase in
negative-transfer days within either animal. Final comparisons retain the
development-selected baseline and use separate source models for the animals.
An unsuccessful final result is retained and cannot trigger retuning.

## Run and reproduce

From the repository root, Python 3.12 and the existing locked environment are
required. Development downloads include only Indy source and development files:

```sh
uv sync --locked
uv run --no-sync python -m calib15 download
```

The current streaming evaluator reproduced the inherited baseline at mean R²
`0.3501094618457011`, with maximum daily R² error `1.64e-14`. The v3 core passed
129 software tests, lint, formatting, and type checks before these batches.
Core commit `d4be944` passed [Ubuntu/Windows CI](https://github.com/Nigmat-future/neurocalib-bench/actions/runs/39159176551).
Software checks do not establish a neuroscience effect.

Search has stopped. Before a final report exists, the following command repeats
all **26 completed current-version configurations and 234 daily rows**, without
adding search events:

```sh
uv run --no-sync python -m calibnext reproduce
```

This writes `studies/calibnext/results/reproduced/development.json`. Scientific
row fields must agree to `1e-7`; elapsed times may
differ. The recorded sanity evidence is already present. Do not rerun `sanity`
after trials start: the command rejects overwriting bound evidence.

To reproduce either historical continuation, use a separate checkout of its
source commit listed above. At those commits, the logs occupy their original
`studies/calibnext` path. Run the same locked setup, development download, and
`python -m calibnext reproduce` there. Current v3 source deliberately rejects
using the old ledgers as a new search.

The [freeze](results/frozen.json) was committed and pushed at `60122c5` before
the final access marker was created. Its 64 scientific input hashes match the
committed bytes. Local verification passed 142 tests, lint, formatting, and type
checks; [Ubuntu/Windows CI passed on the frozen source](https://github.com/Nigmat-future/neurocalib-bench/actions/runs/39159842666).
Final downloads and numerical development reproduction are now running. No final
generalization result is available yet. The original-run final commands are:

```sh
uv run --no-sync python -m calibnext final
uv run --no-sync python -m calibnext reproduce
```

The final access marker and exact frozen hashes are checked before any final
recording or Loco source is downloaded or loaded. After a completed final,
`reproduce` repeats only the frozen final comparisons. Read the
[protocol](PROTOCOL.md) and [research loop](program.md) for the complete contract.
