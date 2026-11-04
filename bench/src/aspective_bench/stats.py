"""Statistics for comparing estimators over many datasets.

The unit of replication is the dataset (Demšar, JMLR 2006): each model gets
one score per dataset, the mean over the repeated cross-validation folds, and
the tests compare those per-dataset scores. Folds of one dataset are not
independent, so they are never pooled into one test.

* ``noninferiority``: the gate. One-sided Wilcoxon signed-rank test of
  H0: theta <= -margin against H1: theta > -margin, where theta is the
  typical difference (candidate - baseline). For the signed-rank test theta
  is the pseudo-median, the Hodges-Lehmann location of the differences;
  for symmetric differences it equals the median. Rejecting H0 is positive
  evidence that the candidate is at most ``margin`` worse than the baseline
  in the typical case; it may still lose on single datasets.
* ``severe_loss_test``: the tail guard. A rank test barely notices a few
  very large losses, so the gate also asks whether, among the datasets where
  the two models differ by more than ``SEVERE`` margins, the candidate loses
  significantly more often than it wins (one-sided exact sign test). Severe
  losses balanced by severe wins are a trade-off the docs can state; severe
  losses with nothing in return are not.
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


SEVERE = 5  # a difference beyond SEVERE * margin counts as a severe win or loss


def severe_loss_test(diff, margin, factor=SEVERE):
    """``(severe wins, severe losses, p)`` for "severe losses outnumber severe wins"."""
    d = np.asarray(diff, dtype=float)
    wins, losses = int(np.sum(d > factor * margin)), int(np.sum(d < -factor * margin))
    if losses == 0:
        return wins, losses, 1.0
    return wins, losses, float(stats.binomtest(losses, wins + losses, 0.5, alternative="greater").pvalue)


def hodges_lehmann(diff):
    """Pseudo-median of the differences: the median of all pairwise (Walsh) averages."""
    d = np.asarray(diff, dtype=float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


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
    pseudo_median: float = np.nan
    severe_wins: int = 0
    severe_losses: int = 0
    p_severe: float = np.nan

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
    severe losses  non-inferior in the typical case, but severe losses
                   (beyond ``SEVERE`` margins) significantly outnumber severe
                   wins at level ``alpha``; this fails
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
    severe_wins, severe_losses, p_severe = severe_loss_test(d, margin)
    if n < min_datasets:
        verdict = "too few datasets"
    elif p_two < alpha and median < 0 and p_ni >= alpha:
        verdict = "inferior"
    elif p_ni < alpha and p_severe < alpha:
        verdict = "severe losses"
    elif p_ni < alpha:
        verdict = "superior" if p_two < alpha and median > 0 else "non-inferior"
    else:
        verdict = "inconclusive"
    return Comparison(
        baseline=baseline_name, n_datasets=n, mean_diff=float(d.mean()), median_diff=median,
        wins=int(np.sum(d > margin)), ties=int(np.sum(np.abs(d) <= margin)),
        losses=int(np.sum(d < -margin)), p_noninferior=p_ni, p_two_sided=p_two, verdict=verdict,
        pseudo_median=hodges_lehmann(d), severe_wins=severe_wins, severe_losses=severe_losses,
        p_severe=p_severe,
    )


def operating_characteristic(pools, margin, n_datasets, shifts, n_sim=1000, alpha=0.05,
                             min_datasets=10, seed=0):
    """Pass rate of the gate for a candidate whose true typical difference is ``shift``.

    ``pools`` are observed per-dataset differences from real comparisons on
    one suite. Each is centred at its pseudo-median, so it keeps the real
    spread, skew and tails between datasets but has a typical difference of
    zero. One simulated comparison draws a pool, resamples ``n_datasets``
    differences from it with replacement, adds ``shift`` and applies
    ``compare``. The result maps each shift to the share of simulations that
    pass: at ``-margin`` and below that is the false-pass rate, at 0 and above
    one minus the miss rate.
    """
    rng = np.random.default_rng(seed)
    centred = [np.asarray(p, float) - hodges_lehmann(p) for p in pools if len(p)]
    zeros = np.zeros(n_datasets)
    out = {}
    for shift in shifts:
        passed = 0
        for _ in range(n_sim):
            pool = centred[rng.integers(len(centred))]
            d = rng.choice(pool, n_datasets) + shift
            passed += compare(d, zeros, "", margin, alpha, min_datasets).passed
        out[shift] = passed / n_sim
    return out


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
