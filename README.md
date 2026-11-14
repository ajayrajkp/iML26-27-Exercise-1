
# OSSI research prototype

This project evaluates the **Operational Structural Simulatability Index (OSSI)** for Decision Tree, Random Forest, and Gradient Boosting classifiers. It runs repeated stratified train/test splits on scikit-learn's built-in Iris, Wine, and Breast Cancer datasets, plus the UCI Bank Marketing dataset. The codebase and generated reports focus exclusively on OSSI.

OSSI estimates structural workload under a stated inspection protocol. It does not prove that a person will understand a model. The budgets are declared protocol choices, not universal limits of human cognition.

## Install

Requires Python 3.9 or newer.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Run multiple experiments

The default is five seeds: 42, 43, 44, 45, and 46.

```bash
python -m ossi.cli --dataset iris --output results/iris
python -m ossi.cli --dataset wine --output results/wine
python -m ossi.cli --dataset breast_cancer --output results/breast_cancer
python -m ossi.cli --dataset bank_marketing --output results/bank_marketing
```

Choose any explicit seed set with `--seeds`, for example:

```bash
python -m ossi.cli --dataset iris --output results/iris_10_runs --seeds 40 41 42 43 44 45 46 47 48 49
```

This produces 3 model runs per seed. Each seed selects a stratified split and initializes the stochastic models. The scikit-learn datasets load locally; Bank Marketing is included as a local, attributed data file. Running the benchmark requires no API key or GPU.

The UCI Bank Marketing dataset is included at `data/raw/bank-full.csv`, with attribution and preprocessing notes in `data/README.md`. Categorical inputs are one-hot encoded after the train/test split; OSSI groups those expanded columns back to their original variables for `F`. The `duration` column is excluded by default because it is only known after the last call and can leak post-contact information. For a local copy at another path, pass `--bank-data /path/to/bank-full.csv`.

## Outputs

- `model_results.csv`: one row per dataset, seed, and model, with accuracy, balanced accuracy, macro F1, and OSSI results. Bank rows additionally include precision, recall, and F1 for subscribers (`yes`).
- `summary_results.csv`: model means, sample standard deviations, majority-class baseline, and moderate/lenient pass rates across seeds.
- `ossi_analysis.csv`: per-run predictive metrics, P, R, F, structural loads, OSSI scores, and budget pass flags. Bank Marketing rows also report precision, recall, and F1 for `yes`.
- `dataset_manifest.json`: dataset provenance, preprocessing settings, and a SHA-256 hash for Bank Marketing.
- `ossi_protocol_comparison.png`: mean OSSI for both protocols with standard-deviation error bars and the score-50 budget boundary.
- `ossi_vs_balanced_accuracy.png`: per-run moderate OSSI and balanced accuracy; this avoids relying on raw accuracy for imbalanced classification.
- `explanations_and_traces.json`: a model summary and one prediction trace per model from the first seed, kept representative to limit file size.

## OSSI formula

For protocol budgets `BP`, `BR`, and `BF`:

```text
L = max(P / BP, R / BR, F / BF)
OSSI = 100 / (1 + L)
```

- `P`: constituent tree count multiplied by mean root-to-leaf path depth across the fixed reference rows. All tree paths count, including class-specific trees in multiclass Gradient Boosting.
- `R`: total number of internal split nodes across the entire model; leaf nodes are excluded.
- `F`: number of distinct input features used in any split.

The runner calculates both protocols:

| Protocol | BP prediction steps | BR split rules | BF features |
|---|---:|---:|---:|
| Moderate | 15 | 30 | 10 |
| Lenient | 30 | 100 | 10 |

An OSSI score of 50 corresponds exactly to the largest normalized workload being 1. Therefore `OSSI >= 50` means all three measured workloads fit the declared protocol budgets. It does **not** establish that people can or will understand the model. The lenient protocol is a sensitivity analysis for a more generous inspection allowance.

### Interpretation caveat

For ensembles, the protocol counts every constituent tree path and assumes the reviewer must then combine their outputs. This makes model-family differences explicit, but actual effort depends on how the model is presented and how votes or boosting contributions are combined. Validate that workload convention with human tasks before making claims about human interpretability. Bank Marketing is class-imbalanced, so raw accuracy is reported alongside balanced accuracy, macro F1, a majority-class baseline, and subscriber (`yes`) precision/recall/F1. The data contains campaign information; this run excludes call duration but remains a retrospective benchmark, not a deployment-ready prediction design.

## Tests

```bash
python -m pytest
```

The suite covers the score formula and boundary, tree/forest/multiclass-boosting aggregation, input and budget validation, score bounds, and a repeated-run output integration check.

## Reproduce on another computer

1. Copy and extract `OSSI_research.zip`.
2. Install Python 3.9 or newer.
3. Open a terminal in the extracted project directory and create a virtual environment with `python -m venv .venv` (Windows may use `py -m venv .venv`).
4. Activate it: `source .venv/bin/activate` on macOS/Linux, or `.venv\\Scripts\\Activate.ps1` in Windows PowerShell.
5. Install with `python -m pip install -e '.[dev]'`. Internet is needed to fetch dependencies unless they are already available locally.
6. Check setup using `python -m pytest`.
7. Run, for example, `python -m ossi.cli --dataset wine --output results/wine`.

Use the same seed list and package versions to reproduce runs as closely as possible. Requirements are minimum versions, not a platform-independent lock file; record Python and dependency versions for formal reporting.

## AI statement

 I created this repository with Codex (Luna 6). After clarifying the scope
with it, it generated the code, tests and README based on my own projects as style references. I
reviewed and edited the result and checked that it runs. I take full responsibility for and
ownership of the content of this repository.


