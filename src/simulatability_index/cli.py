"""Command-line interface for the Simulatability Index project."""

import argparse
import json

from .experiment import run_demo, run_sweep


def main():
    parser = argparse.ArgumentParser(description="Simulatability Index experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    sweep = commands.add_parser("sweep", help="run the complete benchmark")
    sweep.add_argument("--output", default="results/sweep")
    demo = commands.add_parser("demo", help="run three example models on each dataset")
    demo.add_argument("--output", default="results/demo")
    arguments = parser.parse_args()

    if arguments.command == "sweep":
        print(json.dumps(run_sweep(arguments.output), indent=2))
    else:
        print(run_demo(arguments.output).to_string(index=False))


if __name__ == "__main__":
    main()

