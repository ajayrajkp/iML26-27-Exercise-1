"""Run the complete Simulatability Index experiment."""

import json

from simulatability_index.experiment import run_experiment


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))

