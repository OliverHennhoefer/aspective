"""The benchmark gate: every public estimator must be benchmarked and pass.

This reads the committed results in ``benchmarks/results`` and runs no model,
so it is fast and offline. Running the benchmarks themselves is a separate,
local step (``python -m benchmarks run``).
"""

import inspect

import pytest
from sklearn.base import BaseEstimator

import aspective.anomaly
import aspective.discriminant_analysis
import aspective.linear_model
import aspective.naive_bayes
from benchmarks import runner
from benchmarks.protocol import _estimator_classes
from benchmarks.specs import BENCHMARKS, COVERED_BY

MODULES = [aspective.anomaly, aspective.discriminant_analysis, aspective.linear_model,
           aspective.naive_bayes]

PUBLIC_ESTIMATORS = sorted(
    {obj for m in MODULES for obj in vars(m).values()
     if inspect.isclass(obj) and issubclass(obj, BaseEstimator) and obj.__name__ in m.__all__},
    key=lambda c: c.__name__,
)


@pytest.mark.parametrize("cls", PUBLIC_ESTIMATORS, ids=lambda c: c.__name__)
def test_every_estimator_has_a_benchmark(cls):
    covered = set().union(*(_estimator_classes(b.candidate[1]()) for b in BENCHMARKS))
    assert cls in covered or cls.__name__ in COVERED_BY, (
        f"{cls.__name__} has no benchmark: add one to benchmarks/specs.py"
    )


@pytest.mark.parametrize("bench", BENCHMARKS, ids=lambda b: b.name)
def test_benchmark_passes_gate(bench):
    problems = runner.check([bench])
    assert not problems, problems[0]
