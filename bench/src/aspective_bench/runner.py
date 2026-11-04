"""Run benchmarks, summarize them and check the gate."""

import dataclasses
import json
import platform
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import scipy
import sklearn
from joblib import Parallel, delayed

from . import protocol, stats
from .suites import get_suite

# The testbed lives in bench/ next to the package: bench/src/aspective_bench/
# holds the code, bench/results/ and bench/RESULTS.md the committed output.
BENCH_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BENCH_DIR / "results"
ALPHA = 0.05
MIN_DATASETS = 10


def _task(key, args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return key, protocol.evaluate(*args)


def run(benchmarks, n_jobs=-1, quick=False, log=print):
    """Evaluate every model of every benchmark, reusing cached fold scores.

    Returns one summary per benchmark (see ``summarize``). ``quick`` runs
    four datasets with a single repeat to check that a new spec works; its
    results are not meant to be committed.
    """
    plans = []
    for bench in benchmarks:
        names = list(get_suite(bench.suite).datasets)
        if quick:
            bench = dataclasses.replace(bench, n_splits=3, n_repeats=1)
            names = names[:4]
        plans.append((bench, names, protocol.jobs_for(bench, names)))

    todo, seen = [], set()
    for _, _, jobs in plans:
        for key, args in jobs:
            if key not in seen and not protocol.score_cache_path(key).exists():
                seen.add(key)
                todo.append((key, args))
    log(f"{len(todo)} model x dataset evaluations to run, the rest are cached")
    results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(_task)(key, args) for key, args in todo
    )
    for i, (key, result) in enumerate(results, 1):
        protocol.score_cache_path(key).write_text(json.dumps(result))
        failed = f", {len(result['errors'])} failed folds" if result["errors"] else ""
        log(f"[{i}/{len(todo)}] {result['label']} on {result['dataset']}: "
            f"{np.mean(result['scores']):.4f}{failed}")

    summaries = []
    for bench, names, jobs in plans:
        records = [json.loads(protocol.score_cache_path(key).read_text()) for key, _ in jobs]
        summaries.append(summarize(bench, names, records, quick=quick))
    return summaries


def summarize(bench, names, records, quick=False):
    labels = [label for label, _ in bench.models()]
    by = {(r["label"], r["dataset"]): r for r in records}
    means = {label: [float(np.mean(by[label, d]["scores"])) for d in names] for label in labels}
    cand = bench.candidate_label
    margin = bench.effective_margin

    comparisons = [
        stats.compare(means[cand], means[base], base, margin, ALPHA, MIN_DATASETS)
        for base in bench.predecessors
    ]
    matrix = np.array([means[label] for label in labels]).T
    ranks, p_friedman, cd, n_complete = stats.friedman_nemenyi(matrix, ALPHA)
    candidate_failures = [d for d, v in zip(names, means[cand]) if np.isnan(v)]
    passed = not candidate_failures and all(c.passed for c in comparisons)

    return {
        "name": bench.name,
        "fingerprint": None if quick else bench.fingerprint(),
        "quick": quick,
        "suite": bench.suite,
        "metric": bench.metric,
        "higher_is_better": protocol.METRICS[bench.metric][2] > 0,
        "margin": margin,
        "alpha": ALPHA,
        "protocol": bench.protocol_key(),
        "notes": bench.notes,
        "candidate": cand,
        "predecessors": list(bench.predecessors),
        "references": list(bench.references),
        "models": {label: protocol.model_repr(factory()) for label, factory in bench.models()},
        "datasets": names,
        "scores": {label: dict(zip(names, _round(means[label]))) for label in labels},
        "fold_scores": {label: {d: _round(by[label, d]["scores"]) for d in names} for label in labels},
        "seconds": {label: {d: round(float(np.sum(by[label, d]["seconds"])), 3) for d in names}
                    for label in labels},
        "errors": {label: {d: sorted(set(by[label, d]["errors"])) for d in names if by[label, d]["errors"]}
                   for label in labels},
        "comparisons": [dataclasses.asdict(c) | {"passed": c.passed} for c in comparisons],
        "friedman": {"ranks": dict(zip(labels, _round(ranks))), "p": p_friedman, "cd": cd,
                     "n_datasets": n_complete},
        "candidate_failures": candidate_failures,
        "passed": passed,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                        "date": date.today().isoformat()},
    }


def _round(values, digits=6):
    return [None if np.isnan(v) else round(float(v), digits) for v in values]


def save(summary):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{summary['name']}.json"
    path.write_text(json.dumps(summary, indent=1, allow_nan=True) + "\n")
    return path


def load_results():
    """Committed results, with every verdict recomputed by ``assess``."""
    return {p.stem: assess(json.loads(p.read_text())) for p in sorted(RESULTS_DIR.glob("*.json"))}


def _scores(result, label):
    return np.array([np.nan if v is None else v for v in (result["scores"][label][d] for d in result["datasets"])])


def assess(result):
    """Recompute the comparisons and the gate decision from a result's per-dataset scores.

    The committed verdicts are never trusted: the gate is always the current
    statistics applied to the committed scores, so a change to the
    statistics applies to every result without rerunning a model, and a
    hand-edited verdict has no effect.
    """
    cand = result["candidate"]
    comparisons = [
        stats.compare(_scores(result, cand), _scores(result, base), base, result["margin"], ALPHA, MIN_DATASETS)
        for base in result["predecessors"]
    ]
    failures = [d for d, v in zip(result["datasets"], _scores(result, cand)) if np.isnan(v)]
    return result | {
        "alpha": ALPHA,
        "comparisons": [dataclasses.asdict(c) | {"passed": c.passed} for c in comparisons],
        "candidate_failures": failures,
        "passed": not failures and all(c.passed for c in comparisons),
    }


def comparison_diffs(results):
    """Per-dataset differences (candidate - predecessor) of every comparison, grouped by (suite, margin)."""
    pools = {}
    for r in results:
        cand = _scores(r, r["candidate"])
        for base in r["predecessors"]:
            d = cand - _scores(r, base)
            pools.setdefault((r["suite"], r["margin"]), []).append(d[~np.isnan(d)])
    return pools


def check(benchmarks):
    """Problems that block the gate, as a list of strings (empty when all pass)."""
    results = load_results()
    problems = []
    for bench in benchmarks:
        result = results.get(bench.name)
        if result is None:
            problems.append(f"{bench.name}: no committed results; run `uv run aspective-bench run {bench.name}`")
        elif result["fingerprint"] != bench.fingerprint():
            problems.append(f"{bench.name}: results are stale (the spec, suite or the candidate's "
                            f"source changed); rerun `uv run aspective-bench run {bench.name}`")
        elif not result["passed"]:
            why = [f"{c['verdict']} vs {c['baseline']}" for c in result["comparisons"] if not c["passed"]]
            if result["candidate_failures"]:
                why.append(f"errors on {', '.join(result['candidate_failures'])}")
            problems.append(f"{bench.name}: FAIL ({'; '.join(why)})")
    return problems


def main_check(benchmarks):
    problems = check(benchmarks)
    for p in problems:
        print(p)
    print("gate passed" if not problems else f"{len(problems)} benchmark(s) block the gate")
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use `uv run aspective-bench`")
