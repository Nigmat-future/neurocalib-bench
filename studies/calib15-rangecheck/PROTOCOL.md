# Fifteen-second supervised calibration: independent study

This protocol defines a second study. The original six-recording pilot remains an
exploratory antecedent, including its two already observed test dates. None of those
dates is a final target in this study. All new results must be interpreted as
offline natural-movement decoding, not clinical intention decoding or closed-loop
BCI performance. The experiment is allowed to stop without opening the holdout.

## Data and access

`protocol.json` pins the official Zenodo v2 file names, byte sizes and MD5 hashes.
Use M1 threshold crossings pooled within physical channels, causal 20 ms bins,
backward-difference cursor velocity in mm/s, and the original verified MATLAB
reader. Source channels are intersected within an animal only. Missing target
channels are reported and zero-filled. No neuron identity is assumed across days
or animals. The two animals receive separately fitted source models.

Indy uses the two original source dates, 2016-09-30 and 2016-10-05. Nine new dates
from 2016-10-11 through 2016-12-07 are development targets. Five targets dated
2016-12-20 through 2017-01-31 are sealed final dates, later than the latest
previously observed pilot test day. Loco's first three dates, 2017-02-10,
2017-02-13 and 2017-02-14, train its own source model. The following seven dates
are sealed external targets. The manifest is the exact authority for every file.

Before successful development gating and freezing, load behavioral datasets only
for Indy source and development dates. Downloading bytes or checking official
metadata is not behavioral access. Do not inspect Loco source behavior to tune
the shared method. After freezing, Loco source labels may fit that animal's
source model; hyperparameters remain the values selected on Indy.

## Shared source preprocessing

Select common features using only the two Indy source recordings. Evaluate the
full predeclared Cartesian grid: history bins {1, 5, 10}, causal exponential
smoothing time constant {0, 0.06} seconds, and source ridge penalty {0.1, 1, 10}.
These are 18 fixed settings, each evaluated by leaving one source day out and
training on the other. Score the held-out source day at times >= 30.5 seconds.
Choose the highest arithmetic mean of the two macro-axis R² values; break ties
by grid order. Retain every score in `results/source_tuning.json`, and the chosen
setting in `results/features.json`. Refit on all source data before adaptation.

The exponential smoother uses current and past neural bins, restarts after a time
gap and at recording boundaries, and never smooths behavioral labels. Zero
smoothing is the original count feature. Histories never span a gap or session.
All calibration methods share this selected representation and source penalty.
This source-only grid is reported separately from target calibration search.

## Information and scoring

The primary target-label budget is the first **15 elapsed seconds**, including
any startup neural history. All supervised methods receive the same finite-label
prefix rows. No later behavior may select or fit calibration. Score every method
on identical finite-label rows at times >= 30.5 seconds. The fixed final budgets
are 0, 5, 15 and 30 seconds; only 15 seconds may select candidates. At zero budget
all methods return the common frozen source model, including target-only ridge.

Report macro x/y R², each axis's R², RMSE in mm/s, the frozen model's same-day R²,
prefix/scoring row counts, missing channels, feature dimensions and adaptation
wall time. Never replace R² with squared correlation or clip negative scores.
Count negative transfer when adapted R² is strictly below the same-day frozen R².
A negative R² decoding failure and negative transfer are different outcomes.

## Strong baselines and candidate search

Tune both strong supervised baseline families on the nine Indy development
targets, with four fixed penalties {0.01, 0.1, 1, 10} each:

1. Target-only ridge, trained independently on the allowed prefix.
2. Full residual ridge, correcting source predictions from the same prefix.

Choose each family's best by mean development R² at 15 seconds. Choose the
strongest of those two configurations by that same development score. This
baseline remains the primary comparator on every final animal and budget; never
reselect a baseline using test scores. The eight fixed baseline evaluations are
all recorded, and are separate from the maximum of 12 candidate attempts.

The agent-guided candidate surface contains only:

- A six-parameter affine output correction: ridge on the source's two-dimensional
  predictions to fit their target residual, with optional shrinkage toward no
  change. It cannot recover neural information discarded by the source readout.
- A residual readout on source-only PCA directions in standardized neural feature
  space. The number of directions and ridge/shrinkage parameters may be proposed
  in response to development evidence. The PCA never uses target data; dimensions
  are capped at the available source feature count.

Every hypothesis, full configuration, start, success, failure and score is retained
in an append-only JSONL ledger. Try at most **12 distinct candidates**, including
failed or interrupted attempts. A duplicate completed configuration reuses its
record; a failed attempt cannot silently restart. Do not disguise duplicates by
changing inactive parameters. Baselines are a fixed sweep; candidate search is a
sequential hypothesis-and-observation process. Neither is evidence that an
autonomous method discovered a novel neuroscience principle.

After the first target trial starts, scientific code, this protocol, manifest,
agent program, selected features, source-grid results, and numerical dependency
lock are immutable. Fingerprints cover the original reused scientific code too.
Necessary repairs must retain evidence and start a separately identified study;
they cannot silently replace the inputs beneath accumulated scores.

## Development gate, freeze and final confirmation

The gate requires all eight baseline configurations and at least one completed
candidate. Select the highest-scoring candidate that satisfies both conditions:

1. Mean development R² minus the preselected strongest supervised baseline is
   **at least +0.02 absolute R²** at 15 seconds.
2. Its negative-transfer day count is no greater than that baseline's count.

These are practical advancement thresholds, not significance tests. A candidate
that scores highest but violates the harm constraint cannot be selected. If no
candidate passes after the bounded search, save the failed decision and stop.
Do not create a success freeze or open either animal's final targets.

After a pass, create an exclusive freeze binding inputs, complete ledger, selected
candidate and four final configurations: winner, best target-only baseline, best
residual baseline and matched frozen source. Persist a marker bound to this freeze
before reading any final-target behavior. Interrupted final evaluation resumes
the same comparison, without retuning or switching the lock.

Final confirmation uses 15-second scores. First average day-level candidate-minus-
primary-baseline R² separately within each animal, then average the two animal
means equally. Confirmation requires that animal-macro mean >= +0.02, each
animal's mean delta >= 0, and each animal's negative-transfer day count no greater
than the primary baseline's. Report failure honestly if any condition fails. The
5/30-second results, decoding-failure rates, and computational costs are secondary;
they cannot rescue a failed primary confirmation.

There are five Indy and seven Loco final days, but only two animals. Report
individual dates and per-animal summaries. Bins are autocorrelated and cannot be
treated as independent replicates; this design does not establish population-level
or clinical effectiveness. Real-data numerical reproduction must match recorded
scientific values to relative and absolute tolerance 1e-7, excluding wall time.

## Prespecified descriptive failure diagnostics

For the permitted labeled prefix, report the number of occupied equal-width
direction bins out of eight, using samples with speed >= 5 mm/s, and the standard
deviation of prefix speed. These are diagnostic quantities, not a policy to skip
or replace poorly covered prefixes. If direction coverage explains no failures,
report that result. Do not select later fragments using outcomes. Any future
active-sampling experiment must distinguish elapsed waiting time from labeled
data duration and requires another protocol.

## Provenance and scope

The dataset is CC BY 4.0; source: O'Doherty, Cardoso, Makin and Sabes,
`https://zenodo.org/records/3854034`. Raw recordings and processed caches stay
outside Git. Publish the complete search, failed decisions, freeze if achieved,
per-day results, figures and reproduction evidence. Existing FALCON, NoMAD, FDA,
TCLA and earlier adaptive linear decoders preclude a broad novelty claim. A small
linear correction is not itself a new algorithm.
