# Benchmarks

Every estimator in aspective has to show that it is at least competitive with
the vanilla model it claims to improve on. It does not have to win
everywhere. This folder holds the harness that checks that, and its results
are in [`RESULTS.md`](RESULTS.md).

## Protocol

**Suites.** Fixed lists of public datasets, in [`suites.py`](src/aspective_bench/suites.py):

| Suite | Datasets | Source |
|---|---|---|
| `pmlb-classification` | 43 (23 binary, 20 multiclass) | [PMLB](https://github.com/EpistasisLab/pmlb) (Olson et al. 2017; Romano et al. 2022) |
| `pmlb-binary` | the 23 binary ones | PMLB |
| `pmlb-regression` | 25 | PMLB |
| `adbench-classical` | 37 | [ADBench](https://github.com/Minqi824/ADBench) Classical, which includes ODDS (Han et al. 2022) |

Only real-world datasets are used, without missing values, and duplicates
of the same underlying data are removed. Datasets above 5000 rows are
subsampled once with a fixed seed, stratified by class. Files are downloaded
from GitHub on first use and cached in `$ASPECTIVE_BENCH_CACHE` (default
`~/.cache/aspective-bench`) with a SHA-256 check.

**Evaluation.** 2x repeated 5-fold cross-validation, stratified for
classification. Anomaly detectors are fitted without labels on a
contaminated 70% split and scored on the held-out 30%, over 10 stratified
random splits (the ADBench protocol). Small-sample benchmarks (for example
Firth regression) use 10 stratified splits with a fixed number of training
rows. Every model of a benchmark sees exactly the same splits.

**Statistics.** Each model gets one score per dataset, the mean over its
folds. Datasets, not folds, are the unit of replication (Demšar 2006), so
all tests run on the per-dataset scores:

- **Non-inferiority (the gate):** a one-sided Wilcoxon signed-rank test of
  H0 "the candidate is typically worse than the predecessor by more than the
  margin". The margin is 0.01 for accuracy, R² and ROC AUC and 0.005 for the
  Brier score.
- **Two-sided Wilcoxon** for superiority or inferiority.
- **Friedman test with Nemenyi critical difference** over all models of a
  benchmark, including references, for the average-rank summary.

**Verdicts.** `superior` (non-inferior and significantly better),
`non-inferior`, `inconclusive` (neither shown non-inferior nor significantly
worse), `inferior` (significantly worse by more than the margin), and
`too few datasets` (fewer than 10 usable).

**Gate.** A benchmark passes when the candidate runs without error on every
dataset of its suite and is `superior` or `non-inferior` against every
predecessor. References are shown for context only. `inconclusive` fails,
because "not shown to be worse" is not evidence of being competitive.

## Adding a model

1. Add a `Benchmark` to [`specs.py`](src/aspective_bench/specs.py): the candidate, its
   predecessor(s), a suite and a metric. Configure the predecessor like the
   candidate's building blocks (same trees, same binning, same bandwidth),
   so the comparison measures the paper's idea and not a tuning difference.
   ```python
   Benchmark(
       name="my-model",
       candidate=("My model", lambda: MyModel(random_state=0)),
       predecessors={"Vanilla": lambda: Vanilla(random_state=0)},
       suite="pmlb-classification",
       metric="accuracy",
   )
   ```
2. Check that it runs: `uv run aspective-bench run my-model --quick`
   (4 datasets, 3 folds, nothing saved).
3. Run it for real: `uv run aspective-bench run my-model`. Baselines that
   other benchmarks already use come from the score cache. This writes
   `results/my-model.json` and regenerates `RESULTS.md`.
4. Commit both.

The benchmark is a local testbed that lives next to the package: the heavy
runs happen only on your machine, never in CI.
`bench/tests/test_gate.py` runs with the normal test suite (`uv run pytest`,
locally and in CI). It fails if
a public estimator has no benchmark, if a committed result is stale (the spec,
the suite or the candidate's source file changed since it was run), or if a
benchmark does not pass. It reads the committed JSON only, so it is fast.

## Models removed by the gate

The first wave had nine estimators. These failed and were removed after one
retry with the setting their paper recommends, fixed before the retry ran:

| Estimator | Paper | Result |
|---|---|---|
| `RandomRotationRegressor` | Blaser & Fryzlewicz, JMLR 2016 | inferior to random forest: median R² −0.023, worse on 15 of 25 datasets |
| `RandomRotationClassifier` | Blaser & Fryzlewicz, JMLR 2016 | inconclusive vs random forest with min-max scaling (median −0.001) and with rank scaling (median −0.005) |
| `IDKDetector` | Ting et al., KDD 2020 | inconclusive vs Gaussian KDE and one-class SVM with Voronoi cells; the paper's hypersphere kernel was inconclusive at ψ = 16 and worse at ψ = 64 and 256 |
| `RobustKDEDetector` | Kim & Scott, JMLR 2012 | inconclusive vs plain KDE with Scott's bandwidth and with the paper's median nearest-neighbour bandwidth (median difference +0.000 both times) |

Firth's FLIC/FLAC corrections stay as options of `FirthLogisticRegression`,
which passes; at 100 training rows they were not shown better than maximum
likelihood, which the docstring states. The code is in the git history
(commit `aa38a9b` and earlier) if a better configuration turns up.

## Commands

```bash
uv run aspective-bench list            # benchmarks and suites
uv run aspective-bench fetch           # download all datasets
uv run aspective-bench run [names]     # run (all by default), save, report
uv run aspective-bench run name --quick
uv run aspective-bench report          # rebuild RESULTS.md
uv run aspective-bench check           # exit 1 unless the gate passes
```

A full run of all benchmarks takes roughly an hour on four cores.
