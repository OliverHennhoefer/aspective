"""The probabilistic track: scores for predictive distributions.

Point metrics (accuracy, R², ROC AUC) cannot see what a probabilistic model
is good at: an honest spread. These proper scoring rules can.

* ``gaussian_crps`` and ``gaussian_nll`` score a regressor's Gaussian
  predictive distribution. The model follows scikit-learn's convention for
  this, ``predict(X, return_std=True)`` returning the mean and standard
  deviation (as ``GaussianProcessRegressor`` and ``BayesianRidge`` do).
  Before scoring, the target, mean and standard deviation are divided by the
  standard deviation of the training target, so scores are in the same
  units on every dataset and a single margin makes sense.
* ``log_loss`` scores a classifier's ``predict_proba`` over all classes.

CRPS (Gneiting & Raftery, JASA 2007) is the gate metric of choice for
regression: it is proper, in the target's units and robust to a few
overconfident points. NLL is proper too but punishes overconfidence without
bound, so it is better as a second look than as a gate.
"""

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import log_loss as _log_loss
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.utils.validation import check_is_fitted, validate_data

MIN_STD = 1e-6  # in units of the training target's standard deviation


def gaussian_crps(y, dist):
    """Mean CRPS of N(mean, std²) forecasts; lower is better."""
    mean, std = dist
    std = np.maximum(std, MIN_STD)
    z = (y - mean) / std
    return float(np.mean(std * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))))


def gaussian_nll(y, dist):
    """Mean negative log likelihood of N(mean, std²) forecasts; lower is better."""
    mean, std = dist
    std = np.maximum(std, MIN_STD)
    return float(np.mean(0.5 * np.log(2 * np.pi * std**2) + 0.5 * ((y - mean) / std) ** 2))


def log_loss(y, proba_and_classes):
    """Multiclass log loss of ``predict_proba``; lower is better."""
    proba, classes = proba_and_classes
    return float(_log_loss(y, np.clip(proba, 1e-15, 1), labels=classes))


def standardized_dist(model, X, y_train):
    """``(mean, std)`` from ``predict(X, return_std=True)``, in units of ``std(y_train)``."""
    scale = float(np.std(y_train)) or 1.0
    mean, std = model.predict(X, return_std=True)
    return np.asarray(mean, float) / scale, np.asarray(std, float) / scale, scale


class ResidualGaussianRegressor(RegressorMixin, BaseEstimator):
    """A point regressor turned into a Gaussian forecaster: the vanilla baseline.

    The mean is the wrapped estimator's prediction. The standard deviation
    is one constant, the root mean squared error of out-of-fold predictions
    on the training data, so it is honest about the estimator's error
    without any distributional modelling. A probabilistic method has to beat
    this to show that its spread carries information.

    Parameters
    ----------
    estimator : regressor
    cv : int
        Folds for the out-of-fold residuals.
    random_state : int or None
        Shuffles the folds.
    """

    def __init__(self, estimator=None, cv=5, random_state=None):
        self.estimator = estimator
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        X, y = validate_data(self, X, y, y_numeric=True)
        from sklearn.ensemble import GradientBoostingRegressor

        base = GradientBoostingRegressor(random_state=0) if self.estimator is None else self.estimator
        folds = KFold(self.cv, shuffle=True, random_state=self.random_state)
        residuals = y - cross_val_predict(clone(base), X, y, cv=folds)
        self.sigma_ = float(np.sqrt(np.mean(residuals**2)))
        self.estimator_ = clone(base).fit(X, y)
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        mean = self.estimator_.predict(X)
        if return_std:
            return mean, np.full(len(mean), self.sigma_)
        return mean
