import argparse

import pandas as pd

from .datasets import DATASETS


def main(argv=None):
    p = argparse.ArgumentParser(prog="usii", description="USII interpretability experiments")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sweep", help="full complexity sweep + threshold analysis + figures")
    s.add_argument("--output", default="results/sweep")
    s.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=None, help="default: all")
    s.add_argument("--splits", type=int, default=5)
    s.add_argument("--repeats", type=int, default=5)
    s.add_argument("--seed", type=int, default=42)

    a = sub.add_parser("analyze", help="redo threshold/sensitivity/figures from an existing fold_results.csv")
    a.add_argument("--output", default="results/sweep")

    d = sub.add_parser("demo", help="single split with model summaries and prediction traces")
    d.add_argument("--dataset", choices=tuple(DATASETS), default="breast_cancer")
    d.add_argument("--output", default="results/demo")
    d.add_argument("--seed", type=int, default=42)

    args = p.parse_args(argv)
    if args.command == "sweep":
        from .pipeline import run
        _, _, table = run(args.output, args.datasets, args.splits, args.repeats, args.seed)
        from .threshold import primary_summary
        print(primary_summary(table).round(3).to_string(index=False))
    elif args.command == "analyze":
        from .pipeline import analyze
        analyze(pd.read_csv(f"{args.output}/fold_results.csv"), args.output)
    else:
        from .experiment import run as demo
        print(demo(args.output, args.seed, args.dataset).to_string(index=False))


if __name__ == "__main__":
    main()
