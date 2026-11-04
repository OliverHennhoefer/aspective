# Anomaly detection

Unsupervised outlier detectors with PyOD's `BaseDetector` contract: after
`fit(X)` they expose `decision_scores_` (higher is more anomalous),
`threshold_` and `labels_` (0 inlier, 1 outlier), and `decision_function`,
`predict` and `predict_proba` score new data. With `pip install
"aspective[pyod]"` they subclass PyOD's own base class and work inside PyOD's
ensembles and thresholding.

::: aspective.anomaly.RSHashDetector

::: aspective.anomaly.LeSiNNDetector

::: aspective.anomaly.ExtendedIsolationForestDetector
