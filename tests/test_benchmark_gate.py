"""The benchmark gate: every public estimator must be benchmarked and pass.

This reads the committed results in ``benchmarks/results`` and runs no model,
so it is fast and offline. Running the benchmarks themselves is a separate,
manual step (``python -m benchmarks run``, or the "benchmarks" workflow).
"""

import inspect

import pytest
from sklearn.base import BaseEstimator

import aspective.anomaly
import aspective.discriminant_analysis
import aspective.ensemble
import aspective.linear_model
import aspective.naive_bayes
from benchmarks import runner
from benchmarks.protocol import _estimator_classes
from benchmarks.specs import BENCHMARKS, COVERED_BY, PENDING

MODULES = [aspective.anomaly, aspective.discriminant_analysis, aspective.ensemble,
           aspective.linear_model, aspective.naive_bayes]

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


def _gate_cases():
    for b in BENCHMARKS:
        marks = [pytest.mark.xfail(reason=PENDING[b.name], strict=True)] if b.name in PENDING else []
        yield pytest.param(b, id=b.name, marks=marks)


@pytest.mark.parametrize("bench", list(_gate_cases()))
def test_benchmark_passes_gate(bench):
    problems = runner.check([bench])
    assert not problems, problems[0]


@pytest.mark.parametrize("name", sorted(PENDING))
def test_pending_benchmarks_have_fresh_results(name):
    # A pending benchmark may fail the gate, but its results must stay current.
    problems = [p for p in runner.check([b for b in BENCHMARKS if b.name == name]) if "FAIL" not in p]
    assert not problems, problems[0]
