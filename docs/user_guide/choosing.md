# Choosing an estimator

Start from the model you would use anyway, then look up its aspective
variants. Every variant here was shown on public benchmarks to be at least
as good as the vanilla model (see [Benchmarks](../benchmarks.md)); the
*Verdict* column says whether it was also significantly better.

## Classification

| If you would use | Try | When it helps | Verdict |
|---|---|---|---|
| LDA or QDA | [`RegularizedDiscriminantAnalysis`][aspective.discriminant_analysis.RegularizedDiscriminantAnalysis] | Few samples per class relative to the number of features; tune `lam` and `gamma` between LDA, QDA and a ridge-like identity | non-inferior |
| LDA | [`MixtureDiscriminantAnalysis`][aspective.discriminant_analysis.MixtureDiscriminantAnalysis] | Classes made of several clusters, so a single centroid per class is too coarse | non-inferior |
| Naive Bayes | [`AODEClassifier`][aspective.naive_bayes.AODEClassifier] | The default upgrade: no structure search, no tuning, and much of the bias of the independence assumption removed | superior |
| Naive Bayes | [`A2DEClassifier`][aspective.naive_bayes.A2DEClassifier] | Thousands of rows or more; lower bias than AODE, at a cost that grows with the square of the number of features | superior |
| Naive Bayes | [`TANClassifier`][aspective.naive_bayes.TANClassifier], [`KDBClassifier`][aspective.naive_bayes.KDBClassifier] | Correlated features and one readable dependency structure rather than an average of many; KDB's `k` sets how many parents each feature may have and needs more data | superior |
| Naive Bayes | [`HiddenNaiveBayesClassifier`][aspective.naive_bayes.HiddenNaiveBayesClassifier] | Correlated features and one model with AODE's table sizes; each feature's hidden parent weights the others by conditional mutual information | superior |
| Gaussian naive Bayes | [`FlexibleNaiveBayes`][aspective.naive_bayes.FlexibleNaiveBayes] | Continuous features that are skewed or multimodal, which a single Gaussian per class fits badly | superior |
| k-NN | [`DANNClassifier`][aspective.neighbors.DANNClassifier] | Irrelevant or noisy directions near the class boundary that a round neighbourhood wastes neighbours on | non-inferior |
| Random forest | [`RotationForestClassifier`][aspective.ensemble.RotationForestClassifier] | Continuous features with informative linear combinations, where axis-parallel splits need many cuts | non-inferior |
| AdaBoost | [`LogitBoostClassifier`][aspective.ensemble.LogitBoostClassifier] | Noisy labels, or when you need class probabilities; AdaBoost's exponential loss overreacts to misclassified points | non-inferior |
| AdaBoost | [`MultiBoostClassifier`][aspective.ensemble.MultiBoostClassifier] | AdaBoost's variance is the problem; MultiBoost adds wagging to cut it while keeping boosting's bias reduction | non-inferior |
| Logistic regression | [`FirthLogisticRegression`][aspective.linear_model.FirthLogisticRegression] | Small samples, rare events or complete separation, where maximum likelihood is biased or diverges | non-inferior |
| Logistic regression | [`CauchyLogisticRegression`][aspective.linear_model.CauchyLogisticRegression] | A default weakly informative prior that keeps coefficients finite under separation and stabilizes small samples | superior |

## Regression

| If you would use | Try | When it helps | Verdict |
|---|---|---|---|
| Lasso | [`RelaxedLasso`][aspective.linear_model.RelaxedLasso], [`RelaxedLassoCV`][aspective.linear_model.RelaxedLassoCV] | You need a sparse model whose selected coefficients are not over-shrunk | non-inferior |
| Lasso | [`AdaptiveLasso`][aspective.linear_model.AdaptiveLasso], [`AdaptiveLassoCV`][aspective.linear_model.AdaptiveLassoCV] | Consistent variable selection with less shrinkage of large effects (the oracle property) | non-inferior |
| Lasso | [`NonconvexPenaltyRegression`][aspective.linear_model.NonconvexPenaltyRegression], [`NonconvexPenaltyRegressionCV`][aspective.linear_model.NonconvexPenaltyRegressionCV] | MCP or SCAD: nearly unbiased estimates of large coefficients and fewer noise variables | non-inferior |
| Lasso | [`Bolasso`][aspective.linear_model.Bolasso] | Variable selection that holds up under correlated features, by keeping only variables chosen on every bootstrap replicate | non-inferior |
| Lasso | [`L2BoostRegressor`][aspective.linear_model.L2BoostRegressor] | Smoother coefficient paths than the lasso when features are strongly correlated | non-inferior |
| Linear regression or CART | [`MARSRegressor`][aspective.linear_model.MARSRegressor] | Nonlinear effects and interactions, with a model you can still read term by term | superior |
| Linear regression | [`PSplineGAMRegressor`][aspective.linear_model.PSplineGAMRegressor] | Smooth nonlinear effects, one readable curve per feature | non-inferior |
| CART | [`M5ModelTreeRegressor`][aspective.tree.M5ModelTreeRegressor] | Piecewise-linear trends that a constant-leaf tree approximates by a staircase | superior |
| k-NN regression | [`LoessRegressor`][aspective.neighbors.LoessRegressor] | Smooth signals, especially near the edges of the data where neighbour averages are biased | superior |
| Random forest | [`LocalLinearForestRegressor`][aspective.ensemble.LocalLinearForestRegressor] | Smooth signals and extrapolation toward the edges of the data, where averaging forests are biased | non-inferior |

## Anomaly detection

| If you would use | Try | When it helps | Verdict |
|---|---|---|---|
| Isolation Forest | [`ExtendedIsolationForestDetector`][aspective.anomaly.ExtendedIsolationForestDetector] | Clusters that are not axis-aligned, where axis-parallel cuts leave "ghost" regions of low scores | non-inferior |
| Isolation Forest or the k-NN detector | [`LeSiNNDetector`][aspective.anomaly.LeSiNNDetector] | Large data or clustered outliers that mask each other; tiny random subsamples make it fast and robust | non-inferior |
| Isolation Forest or subspace detectors | [`RSHashDetector`][aspective.anomaly.RSHashDetector] | Outliers that show up only in some feature subsets; linear time with randomized subspace grids | non-inferior |
