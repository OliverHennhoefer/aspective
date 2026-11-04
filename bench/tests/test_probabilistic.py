"""Metrics and the baseline of the probabilistic track."""

import numpy as np
import pytest
from scipy import integrate, stats
from sklearn.linear_model import BayesianRidge, LinearRegression

from aspective_bench.probabilistic import (
    ResidualGaussianRegressor,
    gaussian_crps,
    gaussian_nll,
    log_loss,
    standardized_dist,
)
from aspective_bench.protocol import METRICS


@pytest.mark.parametrize("y, mu, sd", [(0.3, 0.0, 1.0), (-2.0, 0.5, 0.7), (4.0, 4.1, 2.5)])
def test_crps_matches_its_integral_definition(y, mu, sd):
    # CRPS = integral of (F(x) - 1{x >= y})^2 dx (Gneiting & Raftery 2007).
    f = stats.norm(mu, sd).cdf
    below = integrate.quad(lambda x: f(x) ** 2, -np.inf, y)[0]
    above = integrate.quad(lambda x: (1 - f(x)) ** 2, y, np.inf)[0]
    assert gaussian_crps(np.array([y]), (np.array([mu]), np.array([sd]))) == pytest.approx(below + above, rel=1e-6)


def test_nll_matches_scipy():
    y, mu, sd = np.array([0.1, 2.0]), np.array([0.0, 1.0]), np.array([1.0, 0.5])
    assert gaussian_nll(y, (mu, sd)) == pytest.approx(-stats.norm(mu, sd).logpdf(y).mean())


def test_scores_reward_honest_spread():
    rng = np.random.RandomState(0)
    y = rng.normal(size=5000)
    mu = np.zeros_like(y)
    honest, overconfident, vague = (mu, np.ones_like(y)), (mu, np.full_like(y, 0.2)), (mu, np.full_like(y, 5.0))
    for score in (gaussian_crps, gaussian_nll):
        assert score(y, honest) < score(y, overconfident)
        assert score(y, honest) < score(y, vague)


def test_log_loss_uses_all_classes():
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    assert log_loss(np.array([0, 1]), (proba, np.array([0, 1, 2]))) == pytest.approx(-np.log([0.7, 0.8]).mean())


def test_residual_gaussian_spread_is_out_of_fold_error():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(2000, 3))
    y = X @ [1.0, -2.0, 0.5] + rng.normal(scale=0.5, size=2000)
    model = ResidualGaussianRegressor(LinearRegression(), random_state=0).fit(X, y)
    mean, std = model.predict(X, return_std=True)
    assert std[0] == pytest.approx(0.5, abs=0.03)
    np.testing.assert_allclose(mean, model.predict(X))


def test_standardized_dist_is_scale_free():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, 2))
    y = X[:, 0] + rng.normal(scale=0.3, size=300)
    scores = []
    for factor in (1.0, 1000.0):
        model = BayesianRidge().fit(X, y * factor)
        mean, std, scale = standardized_dist(model, X, y * factor)
        scores.append(gaussian_crps(y * factor / scale, (mean, std)))
    assert scores[0] == pytest.approx(scores[1], rel=1e-6)


@pytest.mark.parametrize("metric", ["crps", "nll", "log_loss"])
def test_probabilistic_metrics_are_registered_lower_is_better(metric):
    assert METRICS[metric][2] == -1
