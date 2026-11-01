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
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from aspective.anomaly import LeSiNNDetector, RSHashDetector
from aspective.discriminant_analysis import MixtureDiscriminantAnalysis, RegularizedDiscriminantAnalysis
from aspective.linear_model import (
    AdaptiveLassoCV,
    Bolasso,
    CauchyLogisticRegression,
    FirthLogisticRegression,
    L2BoostRegressor,
    RelaxedLassoCV,
)
from aspective.naive_bayes import AODEClassifier, TANClassifier
from aspective.neighbors import LoessRegressor

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


# LOESS and k-NN get five neighbourhood sizes each, so neither is tuned more.
def loess():
    return GridSearchCV(scaled(LoessRegressor()),
                        {"loessregressor__span": [0.02, 0.05, 0.1, 0.25, 0.5]}, cv=3)


def knn_regression():
    return GridSearchCV(scaled(KNeighborsRegressor()),
                        {"kneighborsregressor__n_neighbors": [3, 5, 10, 20, 40]}, cv=3)


# Shared factories, so the same baseline is computed once and reused.
NB_5_BINS = ("Naive Bayes (5 bins)", binned_nb)
ML_LOGISTIC = ("ML logistic", ml_logistic)
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
        predecessors=dict([NB_5_BINS]),
        references={"Gaussian NB": GaussianNB},
        suite="pmlb-classification",
        metric="accuracy",
        notes="Both use the same 5-bin quantile discretization, as in Webb et al. (2005).",
    ),
    Benchmark(
        name="firth",
        candidate=("Firth", firth()),
        predecessors=dict([ML_LOGISTIC]),
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
    Benchmark(
        name="tan",
        candidate=("TAN (5 bins)", lambda: TANClassifier(n_bins=5)),
        predecessors=dict([NB_5_BINS]),
        references={"AODE (5 bins)": lambda: AODEClassifier(n_bins=5)},
        suite="pmlb-classification",
        metric="accuracy",
        notes="Both use the same 5-bin quantile discretization and Laplace smoothing.",
    ),
    Benchmark(
        name="mda",
        candidate=("MDA (tuned)", lambda: GridSearchCV(
            MixtureDiscriminantAnalysis(random_state=0), {"n_subclasses": [1, 2, 3, 5]}, cv=3)),
        predecessors={"LDA": LinearDiscriminantAnalysis},
        references={"QDA": QuadraticDiscriminantAnalysis},
        suite="pmlb-classification",
        metric="accuracy",
        notes="MDA picks the number of subclasses per class by inner 3-fold CV from 1 to 5, as "
              "Hastie & Tibshirani (1996) did; one subclass is LDA.",
    ),
    Benchmark(
        name="cauchy-logistic",
        candidate=("Cauchy prior", CauchyLogisticRegression),
        predecessors=dict([ML_LOGISTIC]),
        references={"Ridge logistic (C=1)": lambda: scaled(LogisticRegression(max_iter=10_000)),
                    "Firth": firth()},
        suite="pmlb-binary",
        metric="brier",
        notes="The Cauchy model applies the paper's own input scaling (sd 0.5, binary inputs "
              "centred); the other models get standardized inputs.",
    ),
    Benchmark(
        name="adaptive-lasso",
        candidate=("Adaptive LassoCV", lambda: scaled(AdaptiveLassoCV(cv=5))),
        predecessors=dict([LASSO_CV]),
        suite="pmlb-regression",
        metric="r2",
        notes="gamma in {0.5, 1, 2} and alpha chosen by 5-fold CV; ridge initial estimator.",
    ),
    Benchmark(
        name="l2boost",
        candidate=("L2Boost (CV)", lambda: scaled(L2BoostRegressor(cv=5))),
        predecessors=dict([LASSO_CV]),
        suite="pmlb-regression",
        metric="r2",
        notes="Componentwise linear least squares, learning rate 0.1, number of steps (up to "
              "2000) by 5-fold CV, as in mboost.",
    ),
    Benchmark(
        name="loess",
        candidate=("LOESS (tuned span)", loess),
        predecessors={"k-NN (tuned k)": knn_regression},
        suite="pmlb-regression",
        metric="r2",
        notes="Local linear fits with tricube weights. Both tune their neighbourhood size over "
              "five values by inner 3-fold CV on standardized inputs.",
    ),
    Benchmark(
        name="lesinn",
        candidate=("LeSiNN", lambda: scaled(LeSiNNDetector(random_state=0))),
        predecessors=dict([ISOLATION_FOREST]),
        references=dict([LOF]),
        suite="adbench-classical",
        metric="roc_auc",
        notes="The paper's defaults: 50 subsamples of 8 points.",
    ),
]

BY_NAME = {b.name: b for b in BENCHMARKS}

# Public estimators whose benchmark is another estimator's, with the reason.
COVERED_BY = {
    "RelaxedLasso": "relaxed-lasso: RelaxedLassoCV is RelaxedLasso with (alpha, phi) chosen by CV",
    "AdaptiveLasso": "adaptive-lasso: AdaptiveLassoCV is AdaptiveLasso with (alpha, gamma) chosen by CV",
}
