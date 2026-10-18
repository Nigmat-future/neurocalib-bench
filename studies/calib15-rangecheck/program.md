# Agent-guided fifteen-second calibration research

Read `PROTOCOL.md` and `protocol.json` before any experiment. The goal is a fair
comparison against tuned supervised baselines at the same fifteen-second prefix
budget, followed by untouched two-animal validation only if development passes.

1. Download and verify Indy source/development files only. Complete synthetic
   correctness and lifecycle checks before the first real target trial.
2. Run the fixed eighteen-setting source-only feature/penalty comparison. Retain
   all scores and the chosen features. Do not choose features using target labels.
3. Run the eight predeclared baseline settings, preserving every full result.
   Identify the best target-only and residual configurations using development.
4. Propose one candidate and state its hypothesis and predicted failure. Start
   with a six-parameter affine correction and a modest source PCA residual model.
   Interpret the observed development response before selecting another config.
5. Keep or discard candidates using development evidence. Try at most twelve
   distinct candidate configurations; crashes consume an attempt. All hypotheses,
   immutable hashes, environmental versions and unsuccessful attempts remain in
   the ledger. No source, scorer, budget, split or metric edits after trial start.
6. A +0.02 mean R² improvement over the strongest tuned supervised baseline and
   no increase in negative-transfer days are both required on development. Stop
   early when the declared gate passes, or when the remaining evidence no longer
   justifies another candidate. Save the decision even if it fails.
7. Freeze only a passing candidate. Download/fit the Loco source model and evaluate
   the exact frozen comparisons on both animals' sealed targets. Do not inspect
   final labels or summaries beforehand. Final access begins with a durable
   freeze-bound marker; only unchanged interrupted work may resume.
8. Retain all final outcomes. Two-animal nondegradation and animal-macro +0.02
   are final confirmation conditions, not information available to development.
   Never reopen search because the final result is disappointing.
9. Recompute the fixed evidence and check scientific values to 1e-7. Report
   software tests separately from biological evidence and model novelty.

The benchmark requires no language-model API. The external coding agent supplies
the hypotheses and commands; the repository supplies the bounded, inspectable
experiment loop. A fixed baseline sweep is identified as a sweep.
