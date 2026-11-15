# Simulatability Index

This project measures the interpretability of tree-based machine-learning models using the **Simulatability Index**.

It evaluates:

- Decision Trees
- Random Forests
- Gradient-Boosted Trees

## Metric

The metric measures three parts of model complexity:

- `P`: prediction steps needed for a typical prediction
- `R`: total number of split rules
- `F`: number of features used by the model

```text
prediction steps = number of trees × average path length
```

Each value is compared with a workload limit:

```text
step load = P / step limit
rule load = R / rule limit
feature load = F / feature limit
```

The largest load is used:

```text
largest load = max(step load, rule load, feature load)
```

The final score is:

```text
Simulatability Index = 100 / (1 + largest load)
```

A higher score means the model is easier to inspect and follow.

## Interpretability Threshold

The primary moderate limits are:

| Measurement | Limit |
|---|---:|
| Prediction steps | 15 |
| Total rules | 30 |
| Used features | 10 |

A model passes the interpretability threshold when:

```text
Simulatability Index >= 50
```

A score of 50 means the largest workload is exactly at its limit.

- Above 50: all workloads are within their limits.
- Equal to 50: at least one workload is exactly at its limit.
- Below 50: at least one workload exceeds its limit.

A lenient backup protocol uses:

| Measurement | Limit |
|---|---:|
| Prediction steps | 30 |
| Total rules | 100 |
| Used features | 10 |

These limits are project choices, not universal limits on human understanding.

## Experiment

The experiment uses:

- 4 datasets: Iris, Wine, Breast Cancer, and UCI Banknote Authentication
- 3 model families
- 3 random train-test splits
- 42 model configurations for each dataset and split
- 504 fitted models for each protocol

The moderate protocol produced:

```text
197 of 504 models passed.
```

The lenient protocol produced:

```text
255 of 504 models passed.
```

A sensitivity analysis tests 27 different combinations of step, rule, and feature limits.

## Project Structure

```text
simulatability_index/
├── data/                   Dataset and attribution
├── presentation/           Presentation outline
├── results/
│   ├── demo/               Small example runs
│   └── sweep/
│       ├── figures/        Result plots
│       ├── moderate_results.csv
│       ├── lenient_results.csv
│       ├── recommendations.csv
│       ├── sensitivity.csv
│       └── summary.json
├── scripts/
│   └── run_experiment.py
├── src/
│   └── simulatability_index/
│       ├── __init__.py
│       ├── cli.py
│       ├── datasets.py
│       ├── experiment.py
│       ├── metric.py
│       └── plots.py
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

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Running the Project

Run the tests:

```bash
python -m pytest
```

Run the complete experiment:

```bash
python -m simulatability_index.cli sweep
```

Run the smaller demonstration:

```bash
python -m simulatability_index.cli demo
```

The complete experiment can also be run with:

```bash
python scripts/run_experiment.py
```

## Output Files

The main results are stored in `results/sweep/`.

- `moderate_results.csv`: results using the moderate limits
- `lenient_results.csv`: results using the lenient limits
- `recommendations.csv`: best overall and best passing models
- `sensitivity.csv`: results under different workload limits
- `summary.json`: main experiment summary
- `figures/`: accuracy and threshold plots

## Limitation

The Simulatability Index is a structural measure. It does not directly measure human understanding, feature meaning, domain knowledge, or explanation quality.

A human study would be needed to confirm whether the selected workload limits match real human performance.

## AI statement

I created this repository with Claude Code (Claude Opus 5.5). After clarifying the scope
with it, it generated the code, tests and README based on my own projects as style references. I
reviewed and edited the result and checked that it runs. I take full responsibility for and
ownership of the content of this repository.