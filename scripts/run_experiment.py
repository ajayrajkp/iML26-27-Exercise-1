"""Run the complete Simulatability Index experiment."""

import json

from simulatability_index.experiment import run_sweep


if __name__ == "__main__":
    print(json.dumps(run_sweep(), indent=2))

