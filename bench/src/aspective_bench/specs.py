"""The benchmark every aspective estimator must pass.

To add a model, append a ``Benchmark`` to ``BENCHMARKS``: the candidate, the
vanilla model(s) it improves on, a suite and a metric. Then run

    uv run aspective-bench run <name>
    uv run aspective-bench report

and commit ``bench/results/<name>.json`` and ``bench/RESULTS.md``.
``bench/tests/test_gate.py`` fails while any public estimator has no
benchmark, a stale result, or a failing verdict. See ``bench/README.md``.

Predecessors are configured like the candidate's own building blocks (same
number of trees, same discretization, same bandwidth rule), so a comparison
measures the paper's idea and not a tuning difference.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from aspective.anomaly import RSHashDetector
from aspective.discriminant_analysis import RegularizedDiscriminantAnalysis
from aspective.linear_model import Bolasso, FirthLogisticRegression, RelaxedLassoCV
from aspective.naive_bayes import AODEClassifier

from .protocol import Benchmark

N_TREES = 200


def scaled(estimator):
    return make_pipeline(StandardScaler(), estimator)


def ml_logistic():
    return scaled(LogisticRegression(penalty=None, max_iter=10_000))


def binned_nb():
    binner = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile",
                              quantile_method="averaged_inverted_cdf")
    return make_pipeline(binner, CategoricalNB(min_categories=5))


def firth(correction="none"):
    return lambda: scaled(FirthLogisticRegression(correction=correction))


# Shared factories, so the same baseline is computed once and reused.
ISOLATION_FOREST = ("Isolation Forest", lambda: scaled(IsolationForest(n_estimators=N_TREES, random_state=0)))
LOF = ("LOF", lambda: scaled(LocalOutlierFactor(novelty=True)))
LASSO_CV = ("LassoCV", lambda: scaled(LassoCV(cv=5)))

BENCHMARKS = [
    Benchmark(
        name="rda",
        candidate=("RDA (tuned)", lambda: GridSearchCV(
            RegularizedDiscriminantAnalysis(),
            {"lam": [0, 0.25, 0.5, 0.75, 1], "gamma": [0, 0.05, 0.25, 0.5]}, cv=3)),
        predecessors={"LDA": LinearDiscriminantAnalysis, "QDA": QuadraticDiscriminantAnalysis},
        references={"LDA (Ledoit-Wolf)": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")},
        suite="pmlb-classification",
        metric="accuracy",
        notes="RDA picks (lambda, gamma) by inner 3-fold CV, as in Friedman (1989); lambda=1, "
              "gamma=0 is LDA and lambda=0, gamma=0 is QDA.",
    ),
    Benchmark(
        name="aode",
        candidate=("AODE (5 bins)", lambda: AODEClassifier(n_bins=5)),
        predecessors={"Naive Bayes (5 bins)": binned_nb},
        references={"Gaussian NB": GaussianNB},
        suite="pmlb-classification",
        metric="accuracy",
        notes="Both use the same 5-bin quantile discretization, as in Webb et al. (2005).",
    ),
    Benchmark(
        name="firth",
        candidate=("Firth", firth()),
        predecessors={"ML logistic": ml_logistic},
        suite="pmlb-binary",
        metric="brier",
        notes="Full-size training folds; features standardized for every model.",
    ),
    Benchmark(
        name="relaxed-lasso",
        candidate=("Relaxed LassoCV", lambda: scaled(RelaxedLassoCV(cv=5))),
        predecessors=dict([LASSO_CV]),
        suite="pmlb-regression",
        metric="r2",
    ),
    Benchmark(
        name="bolasso",
        candidate=("Bolasso", lambda: scaled(Bolasso(random_state=0))),
        predecessors=dict([LASSO_CV]),
        references={"Bolasso-S (0.9)": lambda: scaled(Bolasso(threshold=0.9, random_state=0))},
        suite="pmlb-regression",
        metric="r2",
        notes="Bolasso is mainly a variable-selection method; this checks that its predictions "
              "keep up with the lasso.",
    ),
    Benchmark(
        name="rshash",
        candidate=("RS-Hash", lambda: scaled(RSHashDetector(random_state=0))),
        predecessors=dict([ISOLATION_FOREST]),
        references=dict([LOF]),
        suite="adbench-classical",
        metric="roc_auc",
    ),
]

BY_NAME = {b.name: b for b in BENCHMARKS}

# Public estimators whose benchmark is another estimator's, with the reason.
COVERED_BY = {
    "RelaxedLasso": "relaxed-lasso: RelaxedLassoCV is RelaxedLasso with (alpha, phi) chosen by CV",
}
