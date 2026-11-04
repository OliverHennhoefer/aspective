# Naive Bayes

Semi-naive Bayes classifiers relax the independence assumption of naive Bayes
while keeping its speed and its robustness on small samples. The discrete
ones (AODE, A2DE, TAN, KDB, HNB) treat every distinct value as a category, or
cut continuous features into `n_bins` quantile bins; `FlexibleNaiveBayes`
keeps them continuous.

::: aspective.naive_bayes.AODEClassifier

::: aspective.naive_bayes.A2DEClassifier

::: aspective.naive_bayes.TANClassifier

::: aspective.naive_bayes.KDBClassifier

::: aspective.naive_bayes.HiddenNaiveBayesClassifier

::: aspective.naive_bayes.FlexibleNaiveBayes
