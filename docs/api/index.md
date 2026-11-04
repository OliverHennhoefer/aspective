# API reference

aspective follows scikit-learn's module layout, so each estimator lives where
its vanilla version lives in scikit-learn (or, for anomaly detectors, in
PyOD).

| Module | Estimators |
|---|---|
| [`aspective.discriminant_analysis`](discriminant_analysis.md) | Regularized and mixture discriminant analysis |
| [`aspective.naive_bayes`](naive_bayes.md) | Semi-naive Bayes: AODE, A2DE, TAN, KDB, hidden naive Bayes, flexible naive Bayes |
| [`aspective.neighbors`](neighbors.md) | Adaptive-metric k-NN (DANN) and LOESS |
| [`aspective.ensemble`](ensemble.md) | Rotation Forest, LogitBoost, MultiBoost, local linear forests |
| [`aspective.tree`](tree.md) | M5 model trees |
| [`aspective.linear_model`](linear_model.md) | Firth and Cauchy-prior logistic regression, relaxed, adaptive and nonconvex lasso, Bolasso, L2Boost, MARS, P-spline GAMs |
| [`aspective.anomaly`](anomaly.md) | RS-Hash, LeSiNN, Extended Isolation Forest |

Every supervised estimator passes scikit-learn's `check_estimator`, so it
works with pipelines, `GridSearchCV`, `cross_val_score`, `clone` and the rest
of scikit-learn. Every anomaly detector follows PyOD's `BaseDetector`
contract, and *is* a PyOD detector when PyOD is installed.
