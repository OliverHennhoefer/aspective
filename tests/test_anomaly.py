"""PyOD-contract and detection-quality checks for the anomaly detectors."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from aspective.anomaly import LeSiNNDetector, RSHashDetector
from aspective.anomaly._base import BaseDetector

DETECTORS = [
    RSHashDetector(n_estimators=50, random_state=0),
    LeSiNNDetector(random_state=0),
]


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
    if not isinstance(model, (RSHashDetector, LeSiNNDetector)):
        # These score training points leave-one-out (a point is not its own neighbour),
        # so decision_function on the training data differs from decision_scores_.
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
