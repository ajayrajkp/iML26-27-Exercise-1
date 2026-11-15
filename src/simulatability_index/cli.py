"""Command-line interface for the Simulatability Index project."""

import argparse
import json

from .experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Simulatability Index experiments")
    parser.add_argument("--output", default="results")
    arguments = parser.parse_args()
    print(json.dumps(run_experiment(arguments.output), indent=2))


if __name__ == "__main__":
    main()

