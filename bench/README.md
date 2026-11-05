# Benchmarks

Every estimator in aspective has to show that it is at least competitive with
the vanilla model it claims to improve on. It does not have to win
everywhere. This folder holds the harness that checks that, and its results
are in [`RESULTS.md`](RESULTS.md).

## Protocol

**Suites.** Fixed lists of public datasets, in [`suites.py`](src/aspective_bench/suites.py):

| Suite | Datasets | Source |
|---|---|---|
| `pmlb-classification` | 68 (31 binary, 37 multiclass) | [PMLB](https://github.com/EpistasisLab/pmlb) (Olson et al. 2017; Romano et al. 2022) |
| `pmlb-binary` | the 31 binary ones | PMLB |
| `pmlb-regression` | 31 | PMLB |
| `pmlb-ctr23-regression` | 52 (31 PMLB, 21 OpenML-CTR23) | PMLB and [OpenML-CTR23](https://www.openml.org/search?type=study&study_type=task&id=353) (Fischer, Feurer & Bischl 2023) |
| `adbench-classical` | 46 | [ADBench](https://github.com/Minqi824/ADBench) Classical, which includes ODDS (Han et al. 2022) |

Only real-world datasets are used, without missing values, and duplicates
of the same underlying data are removed. Classification datasets need at
least 10 rows in the smallest class and are dropped when a default decision
tree already reaches 99.5% accuracy, since every model ties on them. The
full rule is in `suites.py`; it was fixed before the added datasets were
run, and the suites change only on purpose, never to rescue a model.
PMLB has no more real regression datasets that meet the rule, so the
regression benchmarks run on `pmlb-ctr23-regression`, which adds the 21
OpenML-CTR23 datasets that meet it. Of CTR23's 35, six were produced by a
simulator, four are already in PMLB, two have missing values and two have
more than 60k rows; `suites.py` names them. OpenML files are pinned by data
id, file id and MD5, categorical features are coded as integers as in PMLB,
and the attributes OpenML marks to ignore are dropped. Datasets above 5000 rows are
subsampled once with a fixed seed, stratified by class. Files are downloaded
from GitHub or openml.org on first use and cached in `$ASPECTIVE_BENCH_CACHE` (default
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
  margin". "Typically" is the pseudo-median of the per-dataset differences
  (their Hodges-Lehmann location), which is what the signed-rank test
  measures. The margin is 0.01 for accuracy, R² and ROC AUC and 0.005 for
  the Brier score.
- **Severe losses (tail guard):** a rank test barely notices a few very
  large losses. Among the datasets where the two models differ by more than
  five margins, a one-sided exact sign test asks whether the candidate loses
  significantly more often than it wins. Severe losses balanced by severe
  wins are a trade-off the docstring can state; severe losses with nothing
  in return fail the gate.
- **Two-sided Wilcoxon** for superiority or inferiority.
- **Friedman test with Nemenyi critical difference** over all models of a
  benchmark, including references, for the average-rank summary.

**Verdicts.** `superior` (non-inferior and significantly better),
`non-inferior`, `severe losses` (non-inferior in the typical case, but
severe losses significantly outnumber severe wins), `inconclusive` (neither
shown non-inferior nor significantly worse), `inferior` (significantly worse
by more than the margin), and `too few datasets` (fewer than 10 usable).

**Gate.** A benchmark passes when the candidate runs without error on every
dataset of its suite and is `superior` or `non-inferior` against every
predecessor. References are shown for context only. `inconclusive` fails,
because "not shown to be worse" is not evidence of being competitive.
Verdicts are never read from the committed files: `check`, the tests and the
report recompute them from the committed per-dataset scores, so a change to
the statistics applies to every result without rerunning a model.

**Leaderboards.** `RESULTS.md` also ranks every model run on a suite
together: the candidates, their predecessors and references, and a fixed
set of standard methods with default settings (linear and ridge models,
CART, k-NN, SVMs, random forest, extra trees, gradient boosting and
histogram gradient boosting; Isolation Forest, LOF, one-class SVM and KDE for
anomaly detection; an elliptic envelope did not finish on InternetAds, whose
1555 features make its robust covariance fit impractical). The standards are defined in
`specs.py`, saved under `results/standards/` and never affect the gate.

**How reliable the gate is.** `RESULTS.md` has a table of the gate's pass
rate for a candidate whose true typical difference is known, simulated per
suite from the committed comparisons (their per-dataset differences,
centred at zero and resampled, so the real spread and tails between
datasets are kept). At one margin worse the gate passes about 6% of
candidates, close to the 5% it is designed for. An exactly equivalent
candidate, though, is still missed about a third of the time (33% on
classification, 30% on regression, 41% on anomaly detection; almost never
on the binary Brier suite). Most of the spread is genuine variation between
datasets, not fold noise (fold noise is 15 to 50% of it), so more CV
repeats do not help; only more datasets do. Growing the suites from
43/25/37 to 68/31/46 datasets lowered the miss rate less than hoped,
because the added datasets also widened the spread. Adding the 21
OpenML-CTR23 datasets to the 31 PMLB ones lowered the regression miss rate
from 34% to 30% and raised the pass rate at half a margin better from 70% to
78%; no regression verdict changed. A candidate that is genuinely better by
half a margin passes 78 to 85% of the time.

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

The second wave added nine candidates and removed these two, which failed
at the settings their paper recommends:

| Estimator | Paper | Result |
|---|---|---|
| `SupervisedPCARegressor` | Bair, Hastie, Paul & Tibshirani, JASA 2006 | inferior to PCR with CV-chosen components: median R² −0.010, worse on 13 of 25 datasets; it is built for many more features than samples, which PMLB does not have |
| `LoOPDetector` | Kriegel, Kröger, Schubert & Zimek, CIKM 2009 | inconclusive vs LOF at the same 20 neighbours (median ROC AUC −0.013, worse on 20 of 37): turning scores into probabilities with `erf` ties many points at 0 and near 1, which costs ranking quality |

The third wave added twenty-two candidates. Fifteen pass; these seven failed
at the settings their paper recommends, fixed before the run:

| Estimator | Paper | Result |
|---|---|---|
| `DARTClassifier` | Rashmi & Gilad-Bachrach, AISTATS 2015 | inconclusive vs gradient boosting with the same 200 depth-3 trees and learning rate 0.1 (median accuracy −0.006, worse on 16 of 43) |
| `DARTRegressor` | Rashmi & Gilad-Bachrach, AISTATS 2015 | inconclusive vs gradient boosting (median R² −0.001, worse on 11 of 25) |
| `SqrtLasso` | Belloni, Chernozhukov & Wang, Biometrika 2011 | inferior to LassoCV: the pivotal penalty over-shrinks, median R² −0.012, worse on 14 of 25 |
| `SCiForestDetector` | Liu, Ting & Zhou, ECML PKDD 2010 | inconclusive vs Isolation Forest (median ROC AUC −0.001, worse on 16 of 37) |
| `INFLODetector` | Jin, Tung, Han & Wang, PAKDD 2006 | inferior to LOF at the same 20 neighbours (median ROC AUC −0.019, worse on 21 of 37) |
| `AntiHubDetector` (AntiHub2) | Radovanović, Nanopoulos & Ivanović, IEEE TKDE 2015 | inferior to the k-NN distance detector (median ROC AUC −0.078, worse on 28 of 37); built for very high-dimensional data, which ADBench Classical mostly is not |
| `LDOFDetector` | Zhang, Hutter & Jin, PAKDD 2009 | inferior to the k-NN distance detector (median ROC AUC −0.121, worse on 27 of 37) |

SCiForest's range check at test time and AntiHub2's choice of the mixing
weight follow secondary descriptions; the full papers could not be checked.

KDEOS (Schubert, Zimek & Kriegel, SDM 2014) was written but not benchmarked:
the exact definition could not be checked against the paper, and the version
built from the description lost to LOF on a planted-outlier sanity check.

Firth's FLIC/FLAC corrections stay as options of `FirthLogisticRegression`,
which passes; at 100 training rows they were not shown better than maximum
likelihood, which the docstring states. The code is in the git history
(commit `aa38a9b` and earlier for the first wave, commit `2aa9e33` for the second, commit `b2bf66c` for the third) if a better configuration turns up.

### Re-test on the larger suites

The smaller suites missed a truly equivalent model a third of the time or
more, so the seven that had come out inconclusive were re-tested once on
the larger suites: restored from git history exactly as last tested, with
no retuning, and the result declared final in advance. All seven fail
again. `DARTClassifier` and `RandomRotationClassifier` are now inferior;
`DARTRegressor`, `SCiForestDetector`, `RobustKDEDetector`, `IDKDetector`
and `LoOPDetector` stay inconclusive. The six that were clearly inferior
were not re-tested.

Every model ever considered, including those dropped before benchmarking
and those left out, is listed with its paper and outcome in
[`docs/models.md`](../docs/models.md), generated from
[`catalog.py`](src/aspective_bench/catalog.py).

## Commands

```bash
uv run aspective-bench list            # benchmarks and suites
uv run aspective-bench fetch           # download all datasets
uv run aspective-bench run [names]     # run (all by default), save, report
uv run aspective-bench run name --quick
uv run aspective-bench standards [suites]  # run the standard methods for the leaderboards
uv run aspective-bench report          # rebuild RESULTS.md
uv run aspective-bench check           # exit 1 unless the gate passes
```

A full run of all benchmarks takes roughly two to three hours on four cores.
