"""The statistics and plumbing of the benchmark harness (offline, fast)."""

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity

from aspective.anomaly import IDKDetector
from benchmarks import stats
from benchmarks.baselines import ScottKDE
from benchmarks.protocol import anomaly_scores
from benchmarks.specs import scaled

rng = np.random.RandomState(0)
BASE = rng.uniform(0.6, 0.9, size=30)
NOISE = rng.normal(0, 0.003, size=30)


def test_identical_models_are_non_inferior():
    c = stats.compare(BASE, BASE, "base", margin=0.01)
    assert c.verdict == "non-inferior" and c.passed
    assert c.ties == 30


def test_small_loss_within_margin_passes():
    c = stats.compare(BASE - 0.004 + NOISE, BASE, "base", margin=0.01)
    assert c.passed


def test_loss_beyond_margin_fails():
    c = stats.compare(BASE - 0.03 + NOISE, BASE, "base", margin=0.01)
    assert c.verdict == "inferior" and not c.passed
    assert c.losses == 30


def test_clear_gain_is_superior():
    c = stats.compare(BASE + 0.02 + NOISE, BASE, "base", margin=0.01)
    assert c.verdict == "superior"


def test_mixed_results_with_large_spread_are_inconclusive():
    # Mean loss of one margin with a spread of four margins: the data can show neither.
    swing = np.random.RandomState(1).normal(-0.01, 0.04, size=30)
    c = stats.compare(BASE + swing, BASE, "base", margin=0.01)
    assert c.verdict == "inconclusive" and not c.passed


def test_too_few_datasets_never_pass():
    c = stats.compare(BASE[:5] + 0.05, BASE[:5], "base", margin=0.01)
    assert c.verdict == "too few datasets" and not c.passed


def test_nan_datasets_are_dropped():
    cand = BASE.copy()
    cand[:3] = np.nan
    assert stats.compare(cand, BASE, "base", margin=0.01).n_datasets == 27


@pytest.mark.parametrize("k, q", [(2, 1.960), (3, 2.343), (4, 2.569), (5, 2.728)])
def test_nemenyi_matches_demsar_table(k, q):
    # q_0.05 from Table 5(a) of Demšar (2006); CD = q * sqrt(k(k+1) / 6N).
    assert stats.nemenyi_cd(k, 20) == pytest.approx(q * np.sqrt(k * (k + 1) / 120), abs=2e-3)


def test_friedman_ranks():
    scores = np.array([[0.9, 0.8, 0.7], [0.8, 0.9, 0.7], [0.9, 0.8, np.nan]])
    ranks, p, cd, n = stats.friedman_nemenyi(scores)
    assert n == 2
    np.testing.assert_allclose(ranks, [1.5, 1.5, 3.0])


def test_scott_kde_matches_sklearn():
    X = rng.normal(size=(200, 3))
    kde = ScottKDE().fit(X)
    reference = KernelDensity(bandwidth=kde.bandwidth_).fit(X)
    np.testing.assert_allclose(kde.score_samples(X[:20]), reference.score_samples(X[:20]))


@pytest.mark.parametrize("model", [scaled(IsolationForest(random_state=0)), scaled(IDKDetector(random_state=0))])
def test_anomaly_scores_are_higher_for_outliers(model):
    X = np.r_[rng.normal(size=(300, 2)), rng.normal(8, 0.5, size=(5, 2))]
    scores = anomaly_scores(model.fit(X), X)
    assert scores[-5:].min() > np.median(scores[:300])
