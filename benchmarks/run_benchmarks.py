"""Quick comparison of each aspective estimator with its vanilla counterpart.

This is a smoke-level benchmark on datasets that ship with scikit-learn or are
simulated, so it runs offline in a few minutes. It is not a replication of
the papers' experiments; those will use the original datasets via OpenML.

Run:  python benchmarks/run_benchmarks.py > benchmarks/RESULTS.md
"""

import warnings

import numpy as np
from sklearn.datasets import (
    load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine,
    make_classification, make_friedman1,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from aspective.anomaly import IDKDetector, RobustKDEDetector, RSHashDetector
from aspective.discriminant_analysis import RegularizedDiscriminantAnalysis
from aspective.ensemble import RandomRotationClassifier, RandomRotationRegressor
from aspective.linear_model import Bolasso, FirthLogisticRegression, RelaxedLassoCV
from aspective.naive_bayes import AODEClassifier

warnings.filterwarnings("ignore")
SEEDS = range(5)


def cv_mean(model, X, y, scoring, cv):
    try:
        return cross_val_score(model, X, y, scoring=scoring, cv=cv, error_score=np.nan).mean()
    except ValueError:  # every fold failed, e.g. LDA with more features than samples
        return np.nan


def table(title, metric, rows, higher_is_better=True):
    print(f"\n### {title}\n\nMetric: {metric} ({'higher' if higher_is_better else 'lower'} is better). "
          "Best value per row in bold.\n")
    models = list(rows[0][1])
    print("| Dataset | " + " | ".join(models) + " |")
    print("|---" * (len(models) + 1) + "|")
    for name, scores in rows:
        vals = np.array([scores[m] for m in models], dtype=float)
        best = np.nanmax(vals) if higher_is_better else np.nanmin(vals)
        # Bold every value within 0.0005 of the best, i.e. equal at the printed precision.
        cells = [f"**{v:.3f}**" if abs(v - best) < 5e-4 else ("n/a" if np.isnan(v) else f"{v:.3f}") for v in vals]
        print(f"| {name} | " + " | ".join(cells) + " |")


def classification_sets():
    sets = {}
    for name, loader in [("iris", load_iris), ("wine", load_wine), ("breast cancer", load_breast_cancer)]:
        X, y = loader(return_X_y=True)
        sets[name] = (X, y)
    X, y = load_digits(return_X_y=True)
    sets["digits"] = (X, y)
    X, y = make_classification(n_samples=60, n_features=40, n_informative=8, n_redundant=10,
                               n_classes=3, n_clusters_per_class=1, random_state=0)
    sets["small n, p=40 (sim.)"] = (X, y)
    rng = np.random.RandomState(0)
    Xo = rng.uniform(-1, 1, size=(800, 6))
    yo = (Xo @ rng.randn(6) > 0).astype(int)
    sets["oblique boundary (sim.)"] = (Xo, yo)
    return sets


def bench_rda(sets):
    rows = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
    grid = {"lam": [0, 0.25, 0.5, 0.75, 1], "gamma": [0, 0.05, 0.25, 0.5]}
    for name, (X, y) in sets.items():
        rda = GridSearchCV(RegularizedDiscriminantAnalysis(), grid, cv=3)
        rows.append((name, {
            "LDA": cv_mean(LinearDiscriminantAnalysis(), X, y, "accuracy", cv),
            "LDA (Ledoit-Wolf)": cv_mean(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), X, y, "accuracy", cv),
            "QDA": cv_mean(QuadraticDiscriminantAnalysis(reg_param=1e-3), X, y, "accuracy", cv),
            "RDA (tuned)": cv_mean(rda, X, y, "accuracy", cv),
        }))
    table("Regularized Discriminant Analysis vs LDA / QDA", "accuracy, 5x2 CV", rows)


def bench_aode(sets):
    rows = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
    binner = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile",
                              quantile_method="averaged_inverted_cdf")
    for name, (X, y) in sets.items():
        rows.append((name, {
            "GaussianNB": cv_mean(GaussianNB(), X, y, "accuracy", cv),
            "CategoricalNB (5 bins)": cv_mean(make_pipeline(binner, CategoricalNB(min_categories=5)), X, y, "accuracy", cv),
            "AODE (5 bins)": cv_mean(AODEClassifier(n_bins=5), X, y, "accuracy", cv),
        }))
    table("AODE vs naive Bayes", "accuracy, 5x2 CV", rows)


def bench_rotation(sets):
    rows = []
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, (X, y) in sets.items():
        rows.append((name, {
            "Random Forest": cv_mean(RandomForestClassifier(n_estimators=200, random_state=0), X, y, "accuracy", cv),
            "Rotated RF": cv_mean(RandomRotationClassifier(n_estimators=200, random_state=0), X, y, "accuracy", cv),
            "Rotated RF (rank)": cv_mean(RandomRotationClassifier(n_estimators=200, scaling="rank", random_state=0), X, y, "accuracy", cv),
        }))
    table("Random Rotation Ensembles vs Random Forest (classification)", "accuracy, 5-fold CV", rows)

    rows = []
    cv = KFold(5, shuffle=True, random_state=0)
    reg_sets = {"diabetes": load_diabetes(return_X_y=True),
                "Friedman #1 (sim.)": make_friedman1(n_samples=500, noise=1.0, random_state=0)}
    for name, (X, y) in reg_sets.items():
        rows.append((name, {
            "Random Forest": cv_mean(RandomForestRegressor(n_estimators=200, max_features=1 / 3, random_state=0), X, y, "r2", cv),
            "Rotated RF": cv_mean(RandomRotationRegressor(n_estimators=200, random_state=0), X, y, "r2", cv),
        }))
    table("Random Rotation Ensembles vs Random Forest (regression)", "R², 5-fold CV", rows)


def bench_firth():
    # Small samples drawn from breast cancer, 5 standardized features, many repeats.
    X_all, y_all = load_breast_cancer(return_X_y=True)
    X_all = StandardScaler().fit_transform(X_all[:, [0, 1, 4, 7, 21]])
    rows = []
    for n_train, rare in [(30, False), (60, False), (200, True)]:
        scores = {m: [] for m in ["ML logistic", "Ridge logistic (C=1)", "Firth", "Firth + FLIC", "Firth + FLAC"]}
        for seed in range(30):
            rng = np.random.RandomState(seed)
            pos, neg = np.flatnonzero(y_all == 0), np.flatnonzero(y_all == 1)  # 0 = malignant
            if rare:
                # Train and test both at a 5% event rate, so calibration is judged fairly.
                n_pos = max(3, n_train // 20)
                pos, neg = rng.permutation(pos), rng.permutation(neg)
                train = np.r_[pos[:n_pos], neg[: n_train - n_pos]]
                rest_neg = neg[n_train - n_pos :]
                test = np.r_[pos[n_pos : n_pos + len(rest_neg) // 19], rest_neg]
            else:
                train = rng.choice(len(y_all), n_train, replace=False)
                test = np.setdiff1d(np.arange(len(y_all)), train)
            y_tr = (y_all[train] == 0).astype(int)
            y_te = (y_all[test] == 0).astype(int)
            if y_tr.min() == y_tr.max():
                continue
            models = {
                "ML logistic": LogisticRegression(C=1e10, max_iter=10_000),
                "Ridge logistic (C=1)": LogisticRegression(C=1.0),
                "Firth": FirthLogisticRegression(),
                "Firth + FLIC": FirthLogisticRegression(correction="flic"),
                "Firth + FLAC": FirthLogisticRegression(correction="flac"),
            }
            for m, model in models.items():
                p = model.fit(X_all[train], y_tr).predict_proba(X_all[test])[:, 1]
                scores[m].append(brier_score_loss(y_te, p))
        label = f"n={n_train}" + (", 5% events" if rare else "")
        rows.append((label, {m: np.mean(v) for m, v in scores.items()}))
    table("Firth logistic regression vs maximum likelihood (breast cancer subsamples, 30 repeats)",
          "Brier score on held-out data", rows, higher_is_better=False)


def sparse_problem(seed, n=100, p=200, k=5, rho=0.5, snr=3.0):
    rng = np.random.RandomState(seed)
    cov = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), cov, size=2 * n)
    beta = np.zeros(p)
    beta[np.sort(rng.choice(p, k, replace=False))] = rng.choice([-1, 1], k) * rng.uniform(1, 2, k)
    signal = X @ beta
    y = signal + rng.randn(2 * n) * np.std(signal) / np.sqrt(snr)
    return X[:n], y[:n], X[n:], y[n:], beta


def bench_lasso_variants():
    rows_mse, rows_f1 = [], []
    for label, kwargs in [("n=100, p=200, SNR=3", {}), ("n=100, p=200, SNR=0.5", {"snr": 0.5}),
                          ("n=200, p=50, SNR=3", {"n": 200, "p": 50})]:
        mse = {m: [] for m in ["LassoCV", "Relaxed LassoCV", "Bolasso", "Bolasso-S (0.9)"]}
        f1 = {m: [] for m in mse}
        for seed in SEEDS:
            X, y, Xt, yt, beta = sparse_problem(seed, **kwargs)
            lasso = LassoCV(cv=5).fit(X, y)
            models = {
                "LassoCV": lasso,
                "Relaxed LassoCV": RelaxedLassoCV(cv=5).fit(X, y),
                "Bolasso": Bolasso(alpha=lasso.alpha_, random_state=seed).fit(X, y),
                "Bolasso-S (0.9)": Bolasso(alpha=lasso.alpha_, threshold=0.9, random_state=seed).fit(X, y),
            }
            for m, model in models.items():
                mse[m].append(np.mean((yt - model.predict(Xt)) ** 2) / np.var(yt))
                f1[m].append(f1_score(beta != 0, model.coef_ != 0))
        rows_mse.append((label, {m: np.mean(v) for m, v in mse.items()}))
        rows_f1.append((label, {m: np.mean(v) for m, v in f1.items()}))
    table("Relaxed Lasso and Bolasso vs Lasso: prediction (simulated, 5 seeds)",
          "test MSE / Var(y)", rows_mse, higher_is_better=False)
    table("Relaxed Lasso and Bolasso vs Lasso: variable selection (simulated, 5 seeds)",
          "F1 of the selected support", rows_f1)


def anomaly_sets(seed):
    rng = np.random.RandomState(seed)
    sets = {}
    X, y = load_breast_cancer(return_X_y=True)
    inl, out = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = rng.choice(out, 20, replace=False)
    sets["breast cancer (malignant as outliers)"] = (X[np.r_[inl, out]], np.r_[np.zeros(len(inl)), np.ones(20)])
    X, y = load_digits(return_X_y=True)
    inl, out = np.flatnonzero(y != 0), rng.choice(np.flatnonzero(y == 0), 40, replace=False)
    sets["digits (zeros as outliers)"] = (X[np.r_[inl, out]], np.r_[np.zeros(len(inl)), np.ones(40)])
    X, y = load_wine(return_X_y=True)
    inl, out = np.flatnonzero(y != 2), rng.choice(np.flatnonzero(y == 2), 10, replace=False)
    sets["wine (class 2 as outliers)"] = (X[np.r_[inl, out]], np.r_[np.zeros(len(inl)), np.ones(10)])
    # Two clusters of different density with local outliers between them.
    a = rng.randn(400, 5) * 0.5
    b = rng.randn(400, 5) * 2 + 8
    local = rng.randn(20, 5) * 0.5 + 2.5
    sets["two densities + local outliers (sim.)"] = (np.vstack([a, b, local]), np.r_[np.zeros(800), np.ones(20)])
    return sets


def bench_anomaly():
    names = list(anomaly_sets(0))
    models = ["Isolation Forest", "LOF", "KDE", "IDK", "RS-Hash", "Robust KDE"]
    results = {n: {m: [] for m in models} for n in names}
    for seed in SEEDS:
        for name, (X, y) in anomaly_sets(seed).items():
            Xs = StandardScaler().fit_transform(X)
            bw = np.mean(Xs.std(axis=0)) * len(Xs) ** (-1 / (Xs.shape[1] + 4))
            scores = {
                "Isolation Forest": -IsolationForest(random_state=seed).fit(Xs).score_samples(Xs),
                "LOF": -LocalOutlierFactor(n_neighbors=20).fit(Xs).negative_outlier_factor_,
                "KDE": -KernelDensity(bandwidth=bw).fit(Xs).score_samples(Xs),
                "IDK": IDKDetector(random_state=seed).fit(Xs).decision_scores_,
                "RS-Hash": RSHashDetector(random_state=seed).fit(Xs).decision_scores_,
                "Robust KDE": RobustKDEDetector().fit(Xs).decision_scores_,
            }
            for m, s in scores.items():
                results[name][m].append(roc_auc_score(y, s))
    rows = [(n, {m: np.mean(v) for m, v in results[n].items()}) for n in names]
    table("Anomaly detectors vs Isolation Forest, LOF and plain KDE (5 seeds, default settings)",
          "ROC AUC on the training data", rows)


if __name__ == "__main__":
    print("# First-wave benchmark results\n")
    print("Generated by `benchmarks/run_benchmarks.py`. Offline smoke benchmark on scikit-learn's bundled "
          "datasets and simulations; it is not a replication of the papers. All models use default "
          "settings unless the column says otherwise.")
    sets = classification_sets()
    bench_rda(sets)
    bench_aode(sets)
    bench_rotation(sets)
    bench_firth()
    bench_lasso_variants()
    bench_anomaly()
