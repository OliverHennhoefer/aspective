# Benchmarks

An estimator only joins aspective if it holds up against the vanilla model
it claims to improve on. It does not have to win everywhere, but the data
have to show it is competitive. The harness lives in
[`bench/`](https://github.com/OliverHennhoefer/aspective/tree/main/bench),
and every committed result is checked in CI.

**[Full results →](https://github.com/OliverHennhoefer/aspective/blob/main/bench/RESULTS.md)**

## Protocol

**Suites.** Fixed lists of real-world public datasets, chosen by a written
rule rather than by hand: [PMLB](https://github.com/EpistasisLab/pmlb) for
classification and regression, and
[ADBench](https://github.com/Minqi824/ADBench) Classical (which includes
ODDS) for anomaly detection. Each suite has dozens of datasets; the
[benchmark README](https://github.com/OliverHennhoefer/aspective/blob/main/bench/README.md)
lists them.

**Evaluation.** 2x repeated 5-fold cross-validation, stratified for
classification. Anomaly detectors are fitted without labels on a
contaminated 70% split and scored on the held-out 30%, over 10 stratified
random splits (the ADBench protocol). Every model of a benchmark sees
exactly the same splits, and the predecessor is configured like the
candidate's building blocks (same trees, same binning, same bandwidth), so
the comparison measures the paper's idea and not a tuning difference.

**Statistics.** Datasets, not folds, are the unit of replication
(Demšar, 2006). Each model gets one score per dataset, and:

- the **gate** is a one-sided Wilcoxon signed-rank test of non-inferiority,
  with a margin of 0.01 for accuracy, R² and ROC AUC and 0.005 for the Brier
  score, at α = 0.05;
- a two-sided Wilcoxon test decides **superior** or **inferior**;
- a Friedman test with Nemenyi critical differences ranks all models of a
  benchmark, including references.

**Verdicts.** `superior` and `non-inferior` pass. `inconclusive` fails,
because "not shown to be worse" is not evidence of being competitive.

## Candidates the gate removed

The gate is not a formality. Published methods that were implemented,
benchmarked at the settings their paper recommends and then failed are
removed, among them the square-root lasso, supervised principal components,
INFLO and LDOF. The
[benchmark README](https://github.com/OliverHennhoefer/aspective/blob/main/bench/README.md#models-removed-by-the-gate)
lists each one with its result.

## Running it yourself

```bash
uv run aspective-bench list            # benchmarks and suites
uv run aspective-bench run name --quick
uv run aspective-bench run [names]     # run, save, report
uv run aspective-bench check           # exit 1 unless the gate passes
```
