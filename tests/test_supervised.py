"""Method-specific checks: each estimator reduces to, or beats, its vanilla version."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsRegressor

from aspective.discriminant_analysis import MixtureDiscriminantAnalysis, RegularizedDiscriminantAnalysis
from aspective.linear_model import (
    AdaptiveLasso,
    AdaptiveLassoCV,
    Bolasso,
    CauchyLogisticRegression,
    FirthLogisticRegression,
    L2BoostRegressor,
    RelaxedLasso,
    RelaxedLassoCV,
)
from aspective.naive_bayes import AODEClassifier, TANClassifier
from aspective.neighbors import LoessRegressor


# --- RDA ---------------------------------------------------------------------

@pytest.fixture
def gaussian_classes():
    X, y = make_classification(n_samples=300, n_features=5, n_informative=4, n_redundant=0,
                               n_classes=3, n_clusters_per_class=1, random_state=0)
    return X, y


def test_rda_lam0_is_qda(gaussian_classes):
    X, y = gaussian_classes
    rda = RegularizedDiscriminantAnalysis(lam=0.0, gamma=0.0).fit(X, y)
    qda = QuadraticDiscriminantAnalysis().fit(X, y)
    # QDA normalizes covariances by n_k - 1, RDA by n_k; probabilities agree closely.
    np.testing.assert_allclose(rda.predict_proba(X), qda.predict_proba(X), atol=0.05)
    assert np.mean(rda.predict(X) == qda.predict(X)) > 0.98


def test_rda_lam1_is_lda(gaussian_classes):
    X, y = gaussian_classes
    rda = RegularizedDiscriminantAnalysis(lam=1.0, gamma=0.0).fit(X, y)
    lda = LinearDiscriminantAnalysis(solver="lsqr").fit(X, y)
    np.testing.assert_allclose(rda.predict_proba(X), lda.predict_proba(X), atol=1e-6)


def test_rda_handles_more_features_than_samples():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 50)
    y = np.repeat([0, 1], 10)
    X[y == 1, :5] += 2
    rda = RegularizedDiscriminantAnalysis(lam=0.5, gamma=0.5).fit(X, y)
    assert np.all(np.isfinite(rda.predict_proba(X)))
    assert rda.score(X, y) > 0.9


# --- AODE --------------------------------------------------------------------

def test_aode_beats_naive_bayes_on_dependent_features():
    # The class is the XOR of two binary features: naive Bayes cannot represent it.
    rng = np.random.RandomState(0)
    X = rng.randint(0, 2, size=(600, 4))
    y = X[:, 0] ^ X[:, 1]
    nb = cross_val_score(CategoricalNB(), X, y, cv=5).mean()
    aode = cross_val_score(AODEClassifier(), X, y, cv=5).mean()
    assert nb < 0.65
    assert aode > 0.95


def test_aode_unseen_categories_and_probabilities():
    X = np.array([[0, 1], [1, 1], [0, 0], [1, 0]] * 5)
    y = np.array([0, 1, 0, 1] * 5)
    model = AODEClassifier().fit(X, y)
    proba = model.predict_proba(np.array([[7, 9], [0, 1]]))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert model.predict(np.array([[0, 1]]))[0] == 0


def test_aode_min_count_falls_back_to_naive_bayes():
    X, y = make_classification(n_samples=200, n_features=4, random_state=0)
    model = AODEClassifier(n_bins=4, min_count=10_000).fit(X, y)
    assert model.score(X, y) > 0.7


# --- Firth -------------------------------------------------------------------

def test_firth_finite_under_complete_separation():
    X = np.array([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = FirthLogisticRegression().fit(X, y)
    assert np.all(np.isfinite(model.coef_))
    assert 0 < model.coef_[0, 0] < 10


def test_firth_score_is_zero_at_solution():
    X, y = make_classification(n_samples=80, n_features=3, n_informative=3, n_redundant=0,
                               random_state=1)
    model = FirthLogisticRegression().fit(X, y)
    design = np.column_stack([np.ones(len(X)), X])
    beta = np.r_[model.intercept_, model.coef_[0]]
    prob = 1 / (1 + np.exp(-design @ beta))
    w = prob * (1 - prob)
    info = design.T @ (w[:, None] * design)
    xs = design * np.sqrt(w)[:, None]
    h = np.einsum("ij,jk,ik->i", xs, np.linalg.inv(info), xs)
    score = design.T @ (y - prob + h * (0.5 - prob))
    np.testing.assert_allclose(score, 0, atol=1e-6)


def test_firth_shrinks_toward_zero_relative_to_ml():
    X, y = make_classification(n_samples=40, n_features=2, n_informative=2, n_redundant=0,
                               class_sep=1.5, random_state=3)
    firth = FirthLogisticRegression().fit(X, y)
    ml = LogisticRegression(C=np.inf, max_iter=10_000).fit(X, y)
    assert np.linalg.norm(firth.coef_) < np.linalg.norm(ml.coef_)


@pytest.mark.parametrize("correction", ["flic", "flac"])
def test_flic_flac_match_event_rate(correction):
    # Rare events: plain Firth overestimates the mean probability, FLIC fixes it exactly.
    rng = np.random.RandomState(0)
    X = rng.randn(400, 3)
    y = (rng.rand(400) < 1 / (1 + np.exp(-(-3 + X[:, 0])))).astype(int)
    firth = FirthLogisticRegression().fit(X, y)
    fixed = FirthLogisticRegression(correction=correction).fit(X, y)
    rate = y.mean()
    assert abs(fixed.predict_proba(X)[:, 1].mean() - rate) < abs(firth.predict_proba(X)[:, 1].mean() - rate)
    if correction == "flic":
        np.testing.assert_allclose(fixed.predict_proba(X)[:, 1].mean(), rate, atol=1e-6)
        np.testing.assert_allclose(fixed.coef_, firth.coef_)


@pytest.mark.parametrize("correction", ["flic", "flac"])
def test_flic_flac_with_uncentered_features(correction):
    # A large offset must not saturate the intercept refit (it used to start at zero).
    X = np.array([[98], [99], [99.5], [100.5], [101], [102]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = FirthLogisticRegression(correction=correction).fit(X, y)
    centered = FirthLogisticRegression(correction=correction).fit(X - 100, y)
    np.testing.assert_allclose(model.predict_proba(X), centered.predict_proba(X - 100), atol=1e-6)
    if correction == "flic":
        np.testing.assert_allclose(model.predict_proba(X)[:, 1].mean(), 0.5, atol=1e-6)


def test_firth_rejects_multiclass():
    with pytest.raises(ValueError, match="Only binary"):
        FirthLogisticRegression().fit(np.random.randn(9, 2), np.arange(9) % 3)


# --- Relaxed Lasso -----------------------------------------------------------

@pytest.fixture
def sparse_problem():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 30)
    coef = np.zeros(30)
    coef[:4] = [3.0, -2.0, 1.5, 1.0]
    y = X @ coef + 0.5 * rng.randn(120)
    return X, y, coef


def test_relaxed_lasso_phi1_is_lasso(sparse_problem):
    X, y, _ = sparse_problem
    relaxed = RelaxedLasso(alpha=0.2, phi=1.0).fit(X, y)
    lasso = Lasso(alpha=0.2).fit(X, y)
    np.testing.assert_allclose(relaxed.coef_, lasso.coef_)


def test_relaxed_lasso_phi0_is_ols_on_support(sparse_problem):
    X, y, _ = sparse_problem
    relaxed = RelaxedLasso(alpha=0.2, phi=0.0).fit(X, y)
    ols = LinearRegression().fit(X[:, relaxed.support_], y)
    np.testing.assert_allclose(relaxed.coef_[relaxed.support_], ols.coef_)


def test_relaxed_lasso_cv_reduces_shrinkage_bias(sparse_problem):
    X, y, coef = sparse_problem
    relaxed = RelaxedLassoCV(cv=5).fit(X, y)
    lasso = Lasso(alpha=relaxed.alpha_).fit(X, y)
    assert np.linalg.norm(relaxed.coef_ - coef) < np.linalg.norm(lasso.coef_ - coef)
    assert relaxed.mse_path_.shape == (50, 5)


# --- Bolasso -----------------------------------------------------------------

def test_bolasso_recovers_support(sparse_problem):
    X, y, coef = sparse_problem
    model = Bolasso(random_state=0).fit(X, y)
    np.testing.assert_array_equal(model.support_, np.flatnonzero(coef))
    np.testing.assert_allclose(model.coef_, coef, atol=0.3)


def test_bolasso_soft_threshold_keeps_superset(sparse_problem):
    X, y, _ = sparse_problem
    strict = Bolasso(alpha=0.05, threshold=1.0, n_bootstrap=32, random_state=0).fit(X, y)
    soft = Bolasso(alpha=0.05, threshold=0.5, n_bootstrap=32, random_state=0).fit(X, y)
    assert set(strict.support_) <= set(soft.support_)


# --- TAN ---------------------------------------------------------------------

def test_tan_learns_a_dependency_naive_bayes_misses():
    # The class is the XOR of features 0 and 1; the tree must link them.
    rng = np.random.RandomState(0)
    X = rng.randint(0, 2, size=(800, 4))
    y = X[:, 0] ^ X[:, 1]
    model = TANClassifier().fit(X, y)
    assert model.parents_[0] == -1
    assert model.parents_[1] == 0
    assert cross_val_score(TANClassifier(), X, y, cv=5).mean() > 0.95
    assert cross_val_score(CategoricalNB(), X, y, cv=5).mean() < 0.65


def test_tan_parents_form_a_tree():
    X, y = make_classification(n_samples=300, n_features=7, random_state=0)
    parents = TANClassifier(n_bins=4).fit(X, y).parents_
    assert np.sum(parents == -1) == 1
    for j in range(len(parents)):  # every feature reaches the root without a cycle
        seen, node = set(), j
        while parents[node] != -1:
            assert node not in seen
            seen.add(node)
            node = parents[node]


def test_tan_unseen_values_and_probabilities():
    X = np.array([[0, 1], [1, 1], [0, 0], [1, 0]] * 5)
    y = np.array([0, 1, 0, 1] * 5)
    proba = TANClassifier().fit(X, y).predict_proba(np.array([[7, 9], [0, 1]]))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


# --- MDA ---------------------------------------------------------------------

def test_mda_one_subclass_is_lda(gaussian_classes):
    X, y = gaussian_classes
    mda = MixtureDiscriminantAnalysis(n_subclasses=1, reg_covar=0).fit(X, y)
    lda = LinearDiscriminantAnalysis(solver="lsqr").fit(X, y)
    np.testing.assert_allclose(mda.predict_proba(X), lda.predict_proba(X), atol=1e-6)


def test_mda_separates_a_multimodal_class():
    # Class 1 sits on both sides of class 0: no linear boundary works.
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(200, 2), rng.randn(100, 2) + [6, 0], rng.randn(100, 2) - [6, 0]])
    y = np.r_[np.zeros(200), np.ones(200)]
    lda = cross_val_score(LinearDiscriminantAnalysis(), X, y, cv=5).mean()
    mda = cross_val_score(MixtureDiscriminantAnalysis(n_subclasses=2, random_state=0), X, y,
                          cv=5).mean()
    assert lda < 0.7
    assert mda > 0.95


# --- Cauchy logistic ---------------------------------------------------------

def test_cauchy_logistic_finite_under_complete_separation():
    X = np.array([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = CauchyLogisticRegression().fit(X, y)
    assert np.all(np.isfinite(model.coef_))
    assert model.predict_proba(X)[:, 1].round().tolist() == y.tolist()


def test_cauchy_logistic_close_to_ml_with_much_data():
    X, y = make_classification(n_samples=4000, n_features=4, n_informative=3, n_redundant=0,
                               random_state=0)
    cauchy = CauchyLogisticRegression().fit(X, y)
    ml = LogisticRegression(penalty=None, max_iter=1000).fit(X, y)
    np.testing.assert_allclose(cauchy.coef_, ml.coef_, rtol=0.02, atol=0.01)


def test_cauchy_logistic_coef_on_original_scale():
    X, y = make_classification(n_samples=300, n_features=3, n_informative=3, n_redundant=0,
                               random_state=1)
    model = CauchyLogisticRegression().fit(X * [1, 100, 0.01] + 5, y)
    eta = (X * [1, 100, 0.01] + 5) @ model.coef_[0] + model.intercept_[0]
    np.testing.assert_allclose(model.decision_function(X * [1, 100, 0.01] + 5), eta)
    assert model.score(X * [1, 100, 0.01] + 5, y) > 0.8


def test_cauchy_logistic_rejects_multiclass():
    X, y = make_classification(n_samples=60, n_classes=3, n_informative=3, random_state=0)
    with pytest.raises(ValueError, match="binary"):
        CauchyLogisticRegression().fit(X, y)


# --- Adaptive lasso ----------------------------------------------------------

def test_adaptive_lasso_recovers_support_with_less_bias(sparse_problem):
    X, y, coef = sparse_problem
    true_support = np.flatnonzero(coef)
    model = AdaptiveLassoCV(cv=5).fit(X, y)
    lasso = LassoCV(cv=5).fit(X, y)
    assert set(true_support) <= set(model.support_)
    assert len(model.support_) <= len(np.flatnonzero(lasso.coef_))


def test_adaptive_lasso_drops_features_with_zero_initial_coef():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 3)
    X[:, 2] = 0.0
    y = X[:, 0] + 0.1 * rng.randn(100)
    model = AdaptiveLasso(alpha=0.01, initial="ols").fit(X, y)
    assert model.coef_[2] == 0
    assert np.isinf(model.weights_[2])


# --- L2 boosting -------------------------------------------------------------

def test_l2boost_converges_to_least_squares():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 5)
    y = X @ [1.0, -2.0, 0.5, 0.0, 3.0] + 0.1 * rng.randn(200)
    model = L2BoostRegressor(max_iter=3000, learning_rate=0.5, cv=None).fit(X, y)
    ols = LinearRegression().fit(X, y)
    np.testing.assert_allclose(model.coef_, ols.coef_, atol=1e-6)
    np.testing.assert_allclose(model.intercept_, ols.intercept_, atol=1e-6)


def test_l2boost_stops_early_on_sparse_problem(sparse_problem):
    X, y, coef = sparse_problem
    true_support = np.flatnonzero(coef)
    model = L2BoostRegressor(cv=5).fit(X, y)
    assert 0 < model.n_iter_ < model.max_iter
    assert set(true_support) <= set(model.support_)
    assert model.cv_mse_.shape == (model.max_iter + 1,)


# --- LOESS -------------------------------------------------------------------

def test_loess_reproduces_a_linear_function_exactly():
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, size=(200, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 1
    pred = LoessRegressor(span=0.2, ridge=0).fit(X, y).predict(X)
    np.testing.assert_allclose(pred, y, atol=1e-8)


def test_loess_less_biased_than_knn_at_the_boundary():
    rng = np.random.RandomState(0)
    X = np.sort(rng.uniform(0, 1, size=(300, 1)), axis=0)
    y = 2 * X[:, 0]
    edge = np.array([[0.0], [1.0]])
    knn = KNeighborsRegressor(n_neighbors=30).fit(X, y).predict(edge)
    loess = LoessRegressor(span=0.1).fit(X, y).predict(edge)
    assert np.all(np.abs(loess - [0, 2]) < 0.01)
    assert np.all(np.abs(knn - [0, 2]) > 0.05)


def test_loess_degree_two_fits_a_parabola():
    X = np.linspace(-1, 1, 101)[:, None]
    y = X[:, 0] ** 2
    pred = LoessRegressor(span=0.3, degree=2, ridge=0).fit(X, y).predict(X)
    np.testing.assert_allclose(pred, y, atol=1e-8)


def test_cauchy_logistic_rejects_a_single_class():
    X = np.random.RandomState(0).randn(20, 2)
    with pytest.raises(ValueError, match="2 classes"):
        CauchyLogisticRegression().fit(X, np.ones(20))
