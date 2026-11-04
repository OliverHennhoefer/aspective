"""Command line: ``uv run aspective-bench {fetch,run,report,check,list}``."""

import argparse
import sys

from . import datasets, report, runner
from .specs import BENCHMARKS, BY_NAME
from .suites import SUITES


def _select(names):
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        sys.exit(f"unknown benchmark(s) {unknown}; see `uv run aspective-bench list`")
    return [BY_NAME[n] for n in names] if names else BENCHMARKS


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aspective-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list benchmarks and suites")
    p = sub.add_parser("fetch", help="download every dataset into the cache")
    p.add_argument("suites", nargs="*")
    p = sub.add_parser("run", help="run benchmarks and save their results")
    p.add_argument("names", nargs="*", help="benchmarks to run (default: all)")
    p.add_argument("-j", "--n-jobs", type=int, default=-1)
    p.add_argument("--quick", action="store_true",
                   help="4 datasets, 3 folds, print only: checks that a new spec works")
    sub.add_parser("report", help="write bench/RESULTS.md from the saved results")
    sub.add_parser("check", help="exit 1 unless every benchmark is fresh and passes the gate")
    args = parser.parse_args(argv)

    if args.command == "list":
        for b in BENCHMARKS:
            print(f"{b.name:22} {b.candidate_label} vs {', '.join(b.predecessors)} on {b.suite} ({b.metric})")
        print()
        for s in SUITES.values():
            print(f"{s.name:22} {s.description}")
    elif args.command == "fetch":
        for name in args.suites or list(SUITES):
            suite = SUITES[name]
            for d in suite.datasets:
                print(name, d, datasets.describe(suite.source, d, suite.task, suite.max_samples))
    elif args.command == "run":
        summaries = runner.run(_select(args.names), n_jobs=args.n_jobs, quick=args.quick)
        for s in summaries:
            if args.quick:
                print(report.section(s))
            else:
                print(f"saved {runner.save(s)}")
        if not args.quick:
            main(["report"])
    elif args.command == "report":
        path = runner.BENCH_DIR / "RESULTS.md"
        results = list(runner.load_results().values())
        path.write_text(report.render(results))
        print(f"wrote {path}")
        docs = runner.BENCH_DIR.parent / "docs" / "models.md"
        docs.write_text(report.render_catalog(results))
        print(f"wrote {docs}")
    elif args.command == "check":
        return runner.main_check(BENCHMARKS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
