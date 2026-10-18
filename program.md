# Agent-guided calibration research

Research question: under a fixed calibration-time budget, can a simple unsupervised
adaptation strategy improve cross-day cursor-velocity decoding over a frozen decoder?
This is a small public-data pilot, not a replication of Neuralink's private clinical work.

## Fixed contract

- Read `protocol.json`, the package source, and the README before proposing experiments.
- Source days train the decoder. Development days select a strategy. Test days remain
  unavailable to search. Never inspect their recordings, labels, summaries or scores.
- Never change the protocol, data, evaluator, package source, dependency lock or scoring
  after the first trial. A necessary repair starts a separately identified research run;
  old scores cannot be mixed into the new run.
- The only search surface is a candidate JSON accepted by `Strategy`. Search `frozen`,
  `center`, `standardize`, and `coral`; `supervised` is a final labeled reference only.
- Objective: mean development R2 across both development sessions and calibration
  budgets of 5, 15, and 30 seconds. The zero-second condition remains in every report.
  Do not select on final-test R2, runtime, visual appeal, or a different aggregate.
- Preprocessing and the fixed channel intersection use only source data. Adaptation
  may use only the explicitly budgeted target prefix. Scoring starts after 30.5 seconds.
- Methods are deterministic closed-form computations. No seed sweep is a hidden search
  dimension. Adding stochastic methods requires a new protocol and fixed seed list.

## Bounded experiment loop

1. Establish a frozen-decoder development baseline first.
2. Read the existing ledger. State one hypothesis and the expected failure mode.
3. Write one candidate JSON. Change only declared strategy parameters.
4. Run one `trial` command with a concise hypothesis. Inspect development results.
5. Retain all `keep`, `discard`, and `crash` records. A repeated canonical strategy
   reuses its result; do not disguise duplicates by editing JSON whitespace or prose.
6. Propose the next hypothesis using the observed development evidence. Prefer a
   simpler method when gains are negligible. A sweep of preset parameters is an
   ordinary search, not evidence of autonomous AI reasoning.
7. Stop after at most 12 distinct attempted candidates, or earlier when the planned
   4–8 informative experiments provide no reason to continue. Crashes consume budget.
   The initially recorded limit cannot be raised during that run.

## Freeze and final evaluation

Run `freeze` once. It selects the development winner and writes an exclusive lock
containing its full strategy, code/protocol/dependency hashes, ledger hash, all five
predeclared default references, and `matched_frozen`: the winner's exact parameters
with adaptation disabled. Use this matched reference to distinguish adaptation gains
from gains caused by selecting ridge strength or history. After freezing, stop search
permanently for this run.

Run `final` only against that lock. A persistent start marker prevents switching locks
after test access begins. If execution fails, resume `final` with unchanged inputs;
completed comparison results are reused. Never delete the marker or lock to retune.

After a complete final report exists, `reproduce` may recompute precisely those frozen
comparisons. It permits a different platform but requires the pinned scientific
dependency versions and identical code, protocol, and lock. It compares scientific
metrics at relative and absolute tolerance `1e-7`, ignores machine timing differences,
and writes a separate `results/reproduced/report.json`. It cannot select new candidates.

The append-only JSONL ledger preserves full starts, strategies, hypotheses, runtime,
checksums, failures, and development results. `results/.active` permits one local
research process. If a process is killed, inspect its PID and ledger before manually
removing that stale process marker; do not remove the final-test start marker.

Publish negative results, data provenance, all attempted candidates, and limitations.
Two development and two test sessions from one animal support a pilot comparison,
not population-level significance, clinical performance, or a broad foundation-model claim.
