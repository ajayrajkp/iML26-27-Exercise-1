# Simulatability Index

This project measures the structural interpretability of Decision Trees, Random Forests, and Gradient-Boosted Trees.

## Metric

The metric uses three measurements:

- `P`: prediction steps for a typical prediction
- `R`: total split rules across all trees
- `F`: number of features used by the model

```text
prediction steps = number of trees × average path length
largest load = max(P / step limit, R / rule limit, F / feature limit)
Simulatability Index = 100 / (1 + largest load)
```

A higher score means lower structural workload.

## Interpretability threshold

The primary moderate limits are:

| Measurement | Limit |
|---|---:|
| Prediction steps | 15 |
| Total rules | 30 |
| Used features | 10 |

A model passes when its Simulatability Index is at least 50. This means all three workloads are within their limits.

The lenient backup protocol uses 30 prediction steps, 100 total rules, and 10 features.

## Experiment

The experiment includes:

- 4 datasets: Iris, Wine, Breast Cancer, and UCI Banknote Authentication
- 3 model families
- 3 random train-test splits
- 42 configurations per dataset and split
- 504 fitted models per protocol
- 27 alternative workload-limit combinations

Under the moderate protocol, 197 of 504 models passed. Under the lenient protocol, 255 of 504 models passed.

## Repository structure

```text
iML26-27-Exercise-1/
├── data/
│   ├── banknote_authentication.csv
│   └── README.md
├── presentation/
│   ├── presentation.pdf
├── results/
│   ├── figures/
│   │   └── accuracy_vs_simulatability.png
│   ├── moderate_results.csv
│   ├── lenient_results.csv
│   ├── recommendations.csv
│   ├── sensitivity.csv
│   └── summary.json
├── scripts/
│   └── run_experiment.py
├── src/simulatability_index/
│   ├── __init__.py
│   ├── cli.py
│   ├── datasets.py
│   ├── experiment.py
│   ├── metric.py
│   └── plots.py
├── tests/
│   ├── test_experiment.py
│   └── test_metric.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

## Run

Run the tests:

```bash
python -m pytest
```

Run the complete experiment:

```bash
python -m simulatability_index.cli
```

The experiment can also be run with:

```bash
python scripts/run_experiment.py
```

All generated files are saved directly in `results/`.

## Output files

- `moderate_results.csv`: primary protocol results
- `lenient_results.csv`: lenient protocol results
- `recommendations.csv`: best overall and best passing configurations
- `sensitivity.csv`: results under different workload limits
- `summary.json`: main experiment summary
- `figures/accuracy_vs_simulatability.png`: accuracy and interpretability plot

## Limitation

The metric measures model structure. Human testing would be needed to confirm the workload limits for a specific audience.

## AI statement

AI-assisted tools supported the code, documentation, and presentation development. The authors reviewed the final project and take responsibility for its content.
