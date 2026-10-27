"""Statistics for comparing estimators over many datasets.

The unit of replication is the dataset (Demšar, JMLR 2006): each model gets
one score per dataset, the mean over the repeated cross-validation folds, and
the tests compare those per-dataset scores. Folds of one dataset are not
independent, so they are never pooled into one test.

* ``noninferiority``: the gate. One-sided Wilcoxon signed-rank test of
  H0: median(candidate - baseline) <= -margin against
  H1: median(candidate - baseline) > -margin. Rejecting H0 is positive
  evidence that the candidate is at most ``margin`` worse than the baseline
  in the typical case; it may still lose on single datasets.
* ``wilcoxon_two_sided``: plain superiority or inferiority.
* ``friedman_nemenyi``: average ranks of all models in a comparison, the
  Friedman test and the Nemenyi critical difference.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _wilcoxon(d, alternative):
    d = np.asarray(d, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    # zsplit keeps exact ties (identical predictions) in the ranking, split
    # evenly between the two sides, instead of silently dropping them.
    return float(stats.wilcoxon(d, zero_method="zsplit", alternative=alternative).pvalue)


def noninferiority(diff, margin):
    """p-value for "candidate is at most ``margin`` worse" (higher is better)."""
    return _wilcoxon(np.asarray(diff) + margin, "greater")


def wilcoxon_two_sided(diff):
    return _wilcoxon(diff, "two-sided")


@dataclass
class Comparison:
    baseline: str
    n_datasets: int
    mean_diff: float
    median_diff: float
    wins: int
    ties: int
    losses: int
    p_noninferior: float
    p_two_sided: float
    verdict: str

    @property
    def passed(self):
        return self.verdict in ("superior", "non-inferior")


def compare(candidate, baseline, baseline_name, margin, alpha=0.05, min_datasets=10):
    """Compare per-dataset scores (higher is better) of a candidate and a baseline.

    ``candidate`` and ``baseline`` are aligned arrays; datasets where either
    is NaN are dropped. "Wins" and "losses" count differences beyond the
    margin; everything within it is a tie.

    Verdicts, from best to worst:

    superior       non-inferior, and better at level ``alpha`` (two-sided)
    non-inferior   at most ``margin`` worse at level ``alpha``; this passes
    inconclusive   neither shown non-inferior nor significantly worse
    inferior       significantly worse at level ``alpha`` (two-sided)
    too few datasets  fewer than ``min_datasets`` usable datasets
    """
    candidate, baseline = np.asarray(candidate, float), np.asarray(baseline, float)
    ok = ~(np.isnan(candidate) | np.isnan(baseline))
    d = candidate[ok] - baseline[ok]
    n = int(ok.sum())
    if n == 0:
        return Comparison(baseline_name, 0, np.nan, np.nan, 0, 0, 0, np.nan, np.nan, "too few datasets")
    p_ni = noninferiority(d, margin)
    p_two = wilcoxon_two_sided(d)
    median = float(np.median(d))
    if n < min_datasets:
        verdict = "too few datasets"
    elif p_two < alpha and median < 0 and p_ni >= alpha:
        verdict = "inferior"
    elif p_ni < alpha:
        verdict = "superior" if p_two < alpha and median > 0 else "non-inferior"
    else:
        verdict = "inconclusive"
    return Comparison(
        baseline=baseline_name, n_datasets=n, mean_diff=float(d.mean()), median_diff=median,
        wins=int(np.sum(d > margin)), ties=int(np.sum(np.abs(d) <= margin)),
        losses=int(np.sum(d < -margin)), p_noninferior=p_ni, p_two_sided=p_two, verdict=verdict,
    )


def average_ranks(scores):
    """Average rank of each column (rank 1 is best, higher score is better)."""
    scores = np.asarray(scores, float)
    return np.mean(stats.rankdata(-scores, axis=1), axis=0)


def nemenyi_cd(k, n, alpha=0.05):
    """Nemenyi critical difference for ``k`` models on ``n`` datasets."""
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def friedman_nemenyi(scores, alpha=0.05):
    """Friedman test and Nemenyi CD on a (datasets x models) score matrix.

    Rows with any NaN are dropped. Returns ``(ranks, p_friedman, cd, n)``;
    the p-value is NaN for fewer than three models.
    """
    scores = np.asarray(scores, float)
    scores = scores[~np.isnan(scores).any(axis=1)]
    n, k = scores.shape
    ranks = average_ranks(scores)
    p = np.nan
    if k >= 3 and n >= 2:
        p = float(stats.friedmanchisquare(*scores.T).pvalue)
    return ranks, p, nemenyi_cd(k, n, alpha), n
