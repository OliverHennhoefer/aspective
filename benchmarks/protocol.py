"""Benchmark definitions and the evaluation protocol.

Every estimator in aspective is registered as a ``Benchmark`` in
``benchmarks/specs.py``: the candidate, the vanilla model(s) it claims to
improve on (``predecessors``), optional extra ``references`` shown in the
report only, a suite and a metric. The gate is applied against every
predecessor; references never affect it.
"""

import hashlib
import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, r2_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from . import datasets
from .suites import get_suite

# Metric name -> (function of (y_true, prediction), what the model must output,
# sign that makes higher better, default non-inferiority margin).
METRICS = {
    "accuracy": (accuracy_score, "predict", 1, 0.01),
    "r2": (r2_score, "predict", 1, 0.01),
    "brier": (brier_score_loss, "proba", -1, 0.005),
    "roc_auc": (roc_auc_score, "anomaly_score", 1, 0.01),
    "average_precision": (average_precision_score, "anomaly_score", 1, 0.01),
}


@dataclass
class Benchmark:
    """One candidate estimator and the vanilla estimators it must keep up with.

    Models are given as zero-argument factories so each fold gets a fresh,
    unfitted estimator. Labels are the names shown in reports.

    Parameters
    ----------
    name : str
        Short identifier, used for the results file and on the command line.
    candidate : (label, factory)
    predecessors : dict of label -> factory
        The vanilla model(s). The candidate must be non-inferior to each.
    suite : str
        Name of a suite in ``benchmarks.suites``.
    metric : str
        Key of ``METRICS``. Scores are oriented so that higher is better.
    references : dict of label -> factory
        Other strong models shown for context; they never affect the gate.
    margin : float or None
        Non-inferiority margin in metric units; None uses the metric default.
    n_splits, n_repeats : int
        Repeated (stratified) k-fold. For anomaly detection, ``n_splits *
        n_repeats`` random stratified 70/30 splits instead (ADBench protocol:
        fit unsupervised on the contaminated 70%, score the held-out 30%).
    train_size : int or None
        If set, ``n_splits * n_repeats`` stratified random splits with this many
        training rows and the rest for testing, to test small-sample claims.
    notes : str
        Why the comparison is set up this way; printed in the report.
    """

    name: str
    candidate: tuple
    predecessors: dict
    suite: str
    metric: str
    references: dict = field(default_factory=dict)
    margin: float | None = None
    n_splits: int = 5
    n_repeats: int = 2
    train_size: int | None = None
    notes: str = ""

    @property
    def candidate_label(self):
        return self.candidate[0]

    @property
    def effective_margin(self):
        return METRICS[self.metric][3] if self.margin is None else self.margin

    def models(self):
        """All (label, factory) pairs: candidate, predecessors, references."""
        return [self.candidate, *self.predecessors.items(), *self.references.items()]

    def protocol_key(self):
        return {"metric": self.metric, "n_splits": self.n_splits, "n_repeats": self.n_repeats,
                "train_size": self.train_size, "suite": self.suite}

    def fingerprint(self):
        """Changes whenever the comparison or the candidate's source changes."""
        suite = get_suite(self.suite)
        parts = {
            "protocol": self.protocol_key(),
            "margin": self.effective_margin,
            "datasets": list(suite.datasets),
            "max_samples": suite.max_samples,
            "models": {label: factory_source(factory) for label, factory in self.models()},
            "source": aspective_source_hash(self.candidate[1]()),
        }
        return _hash(parts)


def factory_source(factory, _depth=0):
    """How a model is specified, independent of the scikit-learn version.

    The repr of a fitted-from-defaults estimator changes when a library changes
    its defaults, which would mark every result stale on another version. The
    source of the factory (plus the constants and helper functions it uses)
    changes only when the spec does.
    """
    if inspect.isclass(factory):
        return f"{factory.__module__}.{factory.__qualname__}"
    if not inspect.isfunction(factory) or _depth > 3:
        return repr(factory)
    names = inspect.getclosurevars(factory)
    used = {}
    for name, value in {**names.nonlocals, **names.globals}.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            used[name] = value
        elif inspect.isfunction(value) and value.__module__ == factory.__module__:
            used[name] = factory_source(value, _depth + 1)
    source = " ".join(inspect.getsource(factory).split())
    return {"source": source, "uses": used}


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def model_repr(estimator):
    with sklearn.config_context(print_changed_only=True):
        return " ".join(repr(estimator).split())


def _estimator_classes(estimator):
    found = {type(estimator)}
    if isinstance(estimator, BaseEstimator):
        for value in estimator.get_params(deep=True).values():
            if isinstance(value, BaseEstimator):
                found |= _estimator_classes(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    step = item[-1] if isinstance(item, tuple) else item
                    if isinstance(step, BaseEstimator):
                        found |= _estimator_classes(step)
    return found


def aspective_source_hash(estimator):
    """Hash of the source files that define the aspective classes the estimator uses.

    Only each class's own module counts, not its bases: the anomaly detectors'
    base class depends on whether PyOD is installed, and the hash must not.
    """
    files = {Path(inspect.getsourcefile(cls)) for cls in _estimator_classes(estimator)
             if cls.__module__.startswith("aspective")}
    if not files:
        return None
    root = Path(__import__("aspective").__file__).parent
    contents = {str(f.relative_to(root)): f.read_text().replace("\r\n", "\n") for f in sorted(files)}
    return _hash(contents)


def model_key(label, estimator, dataset, protocol):
    """Identity of one model x dataset evaluation in the score cache."""
    return _hash({
        "label": label, "repr": model_repr(estimator), "dataset": dataset, "protocol": protocol,
        "max_samples": get_suite(protocol["suite"]).max_samples,
        "aspective": aspective_source_hash(estimator), "sklearn": sklearn.__version__,
        "numpy": np.__version__,
    })


def _final_step(estimator):
    return estimator[-1] if isinstance(estimator, Pipeline) else estimator


def anomaly_scores(estimator, X):
    """Anomaly scores, higher meaning more anomalous, for PyOD-style and sklearn detectors."""
    if hasattr(_final_step(estimator), "decision_scores_"):
        return estimator.decision_function(X)
    return -estimator.score_samples(X)


def splits(task, X, y, n_splits, n_repeats, train_size, seed=0):
    if task == "anomaly":
        return StratifiedShuffleSplit(n_splits * n_repeats, test_size=0.3, random_state=seed).split(X, y)
    if train_size is not None:
        return StratifiedShuffleSplit(n_splits * n_repeats, train_size=min(train_size, len(y) // 2),
                                      random_state=seed).split(X, y)
    if task == "classification":
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(X, y)
    return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(X)


def evaluate(factory, label, suite_name, dataset, metric, n_splits, n_repeats, train_size):
    """Scores of one model on one dataset, one per fold; NaN for a failed fold."""
    suite = get_suite(suite_name)
    X, y = datasets.load(suite.source, dataset, suite.task, suite.max_samples)
    func, output, sign, _ = METRICS[metric]
    scores, seconds, errors = [], [], []
    for train, test in splits(suite.task, X, y, n_splits, n_repeats, train_size):
        model = clone(factory())
        start = time.perf_counter()
        try:
            if suite.task == "anomaly":
                model.fit(X[train])
                pred = anomaly_scores(model, X[test])
            else:
                model.fit(X[train], y[train])
                if output == "proba":
                    pred = model.predict_proba(X[test])[:, 1]
                else:
                    pred = model.predict(X[test])
            if not np.all(np.isfinite(pred)):
                raise FloatingPointError("non-finite predictions")
            scores.append(sign * float(func(y[test], pred)))
        except Exception as exc:  # recorded, so one failure does not stop a run
            scores.append(float("nan"))
            errors.append(f"{type(exc).__name__}: {exc}".splitlines()[0][:300])
        seconds.append(time.perf_counter() - start)
    return {"label": label, "dataset": dataset, "scores": scores, "seconds": seconds, "errors": errors}


def score_cache_path(key):
    path = datasets.cache_dir() / "scores" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def jobs_for(benchmark, dataset_names):
    """(cache key, evaluate-arguments) for every model x dataset of a benchmark."""
    protocol = benchmark.protocol_key()
    out = []
    for label, factory in benchmark.models():
        for dataset in dataset_names:
            key = model_key(label, factory(), dataset, protocol)
            args = (factory, label, benchmark.suite, dataset, benchmark.metric,
                    benchmark.n_splits, benchmark.n_repeats, benchmark.train_size)
            out.append((key, args))
    return out
