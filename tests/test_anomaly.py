"""PyOD-contract and detection-quality checks for the anomaly detectors."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import IsolationForest

from aspective.anomaly import (
    AntiHubDetector,
    ExtendedIsolationForestDetector,
    INFLODetector,
    LDOFDetector,
    LeSiNNDetector,
    RSHashDetector,
    SCiForestDetector,
)
from aspective.anomaly._base import BaseDetector

DETECTORS = [
    RSHashDetector(n_estimators=50, random_state=0),
    LeSiNNDetector(random_state=0),
    ExtendedIsolationForestDetector(n_estimators=50, random_state=0),
    SCiForestDetector(n_estimators=50, random_state=0),
    INFLODetector(),
    AntiHubDetector(),
    LDOFDetector(),
]
# Detectors that score training points leave-one-out (a point is not its own neighbour),
# so decision_function on the training data differs from decision_scores_.
LEAVE_ONE_OUT = (RSHashDetector, LeSiNNDetector, INFLODetector, AntiHubDetector, LDOFDetector)


@pytest.fixture
def planted():
    rng = np.random.RandomState(42)
    inliers = rng.randn(300, 4)
    outliers = rng.uniform(-6, 6, size=(20, 4))
    outliers = outliers[np.linalg.norm(outliers, axis=1) > 4.5]
    X = np.vstack([inliers, outliers])
    y = np.r_[np.zeros(len(inliers)), np.ones(len(outliers))]
    return X, y


@pytest.mark.parametrize("detector", DETECTORS, ids=repr)
def test_pyod_contract(detector, planted):
    X, _ = planted
    model = clone(detector)
    assert isinstance(model, BaseDetector)
    model.fit(X)
    for attr in ("decision_scores_", "threshold_", "labels_"):
        assert hasattr(model, attr)
    assert model.decision_scores_.shape == (len(X),)
    assert set(np.unique(model.labels_)) <= {0, 1}
    assert abs(model.labels_.mean() - model.contamination) < 0.03
    if not isinstance(model, LEAVE_ONE_OUT):
        np.testing.assert_allclose(model.decision_function(X), model.decision_scores_)
        np.testing.assert_array_equal(model.predict(X), model.labels_)
    assert set(np.unique(model.predict(X))) <= {0, 1}
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("detector", DETECTORS, ids=repr)
def test_detects_planted_outliers(detector, planted):
    X, y = planted
    model = clone(detector).fit(X)
    assert roc_auc_score(y, model.decision_scores_) > 0.95


@pytest.mark.parametrize("detector", DETECTORS, ids=repr)
def test_get_set_params_roundtrip(detector):
    params = detector.get_params()
    assert params["contamination"] == 0.1
    assert clone(detector).set_params(contamination=0.2).contamination == 0.2


@pytest.mark.parametrize("detector", DETECTORS, ids=repr)
def test_feature_count_is_checked(detector, planted):
    X, _ = planted
    model = clone(detector).fit(X)
    with pytest.raises(ValueError):
        model.decision_function(X[:, :2])


@pytest.mark.parametrize("detector", DETECTORS, ids=repr)
def test_integer_input_matches_float_input(detector):
    # uint8 arithmetic used to wrap around in the distance computations.
    X = np.random.RandomState(0).randint(0, 50, size=(200, 3)).astype(np.uint8)
    ints = clone(detector).fit(X).decision_scores_
    floats = clone(detector).fit(X.astype(float)).decision_scores_
    np.testing.assert_allclose(ints, floats)


def test_rshash_is_reproducible(planted):
    X, _ = planted
    a = RSHashDetector(random_state=3).fit(X).decision_scores_
    b = RSHashDetector(random_state=3).fit(X).decision_scores_
    np.testing.assert_array_equal(a, b)


def test_works_inside_pyod_ensembles(planted):
    pytest.importorskip("pyod")
    from pyod.models.lscp import LSCP

    X, y = planted
    members = [RSHashDetector(n_estimators=50, random_state=seed) for seed in range(3)]
    ensemble = LSCP(members, random_state=0).fit(X)
    assert roc_auc_score(y, ensemble.decision_scores_) > 0.9


@pytest.mark.parametrize("detector", DETECTORS, ids=repr)
def test_pyod_extras(detector, planted):
    pytest.importorskip("pyod")
    X, _ = planted
    model = clone(detector).fit(X)
    proba = model.predict_proba(X, method="unify")
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    labels, confidence = model.predict(X, return_confidence=True)
    assert labels.shape == confidence.shape == (len(X),)


@pytest.mark.parametrize("n_samples, max_samples", [(2, 8), (50, 1)])
def test_lesinn_training_scores_finite_with_singleton_subsamples(n_samples, max_samples):
    X = np.random.RandomState(0).randn(n_samples, 3)
    model = LeSiNNDetector(max_samples=max_samples, random_state=0).fit(X)
    assert np.all(np.isfinite(model.decision_scores_))
    assert np.isfinite(model.threshold_)


# --- Isolation-forest variants -----------------------------------------------

def test_eif_extension_level_zero_cuts_one_axis(planted):
    X, _ = planted
    model = ExtendedIsolationForestDetector(n_estimators=5, extension_level=0, random_state=0)
    model.fit(X)
    for tree in model.estimators_:
        internal = tree.left != -1
        assert np.all(np.count_nonzero(tree.normal[internal], axis=1) == 1)


def test_eif_scores_are_rotation_symmetric_around_a_blob():
    # On a round blob, Isolation Forest scores points on a circle differently along the
    # axes and the diagonals (its star-shaped score map); oblique cuts remove that.
    rng = np.random.RandomState(0)
    X = rng.randn(2000, 2)
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    circle = 2.5 * np.column_stack([np.cos(angles), np.sin(angles)])
    eif = ExtendedIsolationForestDetector(n_estimators=300, random_state=0).fit(X)
    iso = IsolationForest(n_estimators=300, random_state=0).fit(X)
    assert np.std(eif.decision_function(circle)) < np.std(iso.score_samples(circle))


def test_sciforest_isolates_a_clustered_anomaly():
    # A tight cluster of 30 anomalies next to the normal data: random cuts rarely separate it.
    rng = np.random.RandomState(0)
    normal = rng.randn(1000, 2)
    cluster = rng.randn(30, 2) * 0.05 + [3.5, 0]
    X = np.vstack([normal, cluster])
    y = np.r_[np.zeros(1000), np.ones(30)]
    sci = SCiForestDetector(n_estimators=100, random_state=0).fit(X)
    iso = IsolationForest(n_estimators=100, random_state=0).fit(X)
    auc_sci = roc_auc_score(y, sci.decision_scores_)
    assert auc_sci > 0.95
    assert auc_sci > roc_auc_score(y, -iso.score_samples(X))


def test_sciforest_range_check_isolates_far_points(planted):
    X, _ = planted
    model = SCiForestDetector(n_estimators=20, random_state=0).fit(X)
    far = np.full((1, X.shape[1]), 100.0)
    assert model.decision_function(far)[0] > model.decision_scores_.max()


# --- Nearest-neighbour detectors ---------------------------------------------

def test_inflo_is_one_on_a_regular_grid():
    g = np.arange(12.0)
    X = np.array([[a, b] for a in g for b in g])
    scores = INFLODetector(n_neighbors=4).fit(X).decision_scores_
    interior = (X.min(axis=1) >= 2) & (X.max(axis=1) <= 9)
    np.testing.assert_allclose(scores[interior], 1.0, atol=0.05)


def test_antihub_counts_and_alpha():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(300, 5), [[8, 8, 8, 8, 8]]])
    model = AntiHubDetector(n_neighbors=10).fit(X)
    assert 0 <= model.alpha_ <= 1
    assert model._counts[-1] == 0  # nobody counts the far point among their neighbours
    assert model._counts.sum() == 10 * len(X)
    rank = np.sum(model.decision_scores_ >= model.decision_scores_[-1])
    assert rank <= 0.05 * len(X)


def test_ldof_is_scale_invariant(planted):
    X, _ = planted
    a = LDOFDetector().fit(X).decision_scores_
    b = LDOFDetector().fit(7.5 * X).decision_scores_
    np.testing.assert_allclose(a, b, rtol=1e-8)


def test_ldof_is_below_one_inside_a_cluster():
    # Inside uniform data a point sits among its neighbours, so its mean distance to
    # them is smaller than their mean distance to each other.
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(2000, 2))
    scores = LDOFDetector(n_neighbors=20).fit(X).decision_scores_
    interior = np.all((X > 0.1) & (X < 0.9), axis=1)
    assert np.median(scores[interior]) < 1
