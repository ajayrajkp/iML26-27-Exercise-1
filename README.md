
# USII interpretability research
Develops a structural interpretability metric for tree-based models (the **Unified Structural
Interpretability Index, USII**), applies it *within* three model classes (decision tree, random
forest, gradient-boosted trees) across five datasets, and investigates a **threshold of
interpretability**. No API keys, downloads or GPU are needed; LLM evaluation is out of scope.

## Setup and run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m pytest                                        # 22 tests

python -m usii.cli sweep                                # full study -> results/sweep  (~6 min on 1 CPU)
python -m usii.cli analyze                              # redo analysis/figures from saved fold_results.csv
python -m usii.cli demo --dataset wine                  # one split: model summaries + prediction traces
```

`sweep` caches each finished dataset in `results/sweep/parts/`, so an interrupted run resumes
without refitting. Options: `--datasets iris wine ...`, `--splits`, `--repeats`, `--seed`.

## The metric

`USII = 100 * (1 - (0.30*DPC + 0.30*BC + 0.20*FD + 0.20*EC))`, all terms clipped to [0, 1]:

| Term | Meaning | Normalisation |
|---|---|---|
| DPC | mean root-to-leaf edge depth over fixed reference rows | / 8 |
| BC | mean internal nodes per tree | / (2^8 - 1) |
| FD | fraction of input columns used by at least one split | / n_features |
| EC | number of trees (multiclass boosting: one per class per stage) | / 300 |

Higher USII = less structure to take in. It is meant for ranking models **of the same class**; EC makes
multiclass boosting look larger than binary boosting, so do not compare USII across such settings. Weights and
constants are hypotheses, tested in the sensitivity analysis. (`MAX_TREES` was raised from 100 in the first
prototype to 300 so that no configuration in the sweeps saturates.)

**Simulation cost (SC)** is a second, independent quantity: the number of split conditions a person must evaluate to
reproduce one prediction by hand (sum of path lengths over all trees). It is the cognitive-load criterion for the threshold.

## Threshold of interpretability

A model counts as *interpretable enough* if **SC <= B** (primary B = 9, the upper end of the "7 ± 2" working-memory
heuristic; B in {5, 7, 9, 15, 25} is reported) and as *accurate enough* if its CV accuracy is within one standard error
of the best configuration of the same family on the same dataset (primary; tolerances of 0.01/0.02/0.05 also reported).
The analysis answers: (1) is there any configuration satisfying both (**feasible**)? (2) what is the cheapest
accurate model (`min_sim_cost_accurate`) and the most interpretable accurate model, the **knee**? (3) which USII
value separates models within the cognitive budget from the rest (Youden-optimal `usii_threshold`, with `auc_usii_vs_budget`)?

Setup: 64 configurations per dataset (DT depth 1-8; RF 1-100 trees x depth 2-8; GB 1-100 stages x depth 1-3,
learning rate 0.3), 5x5 repeated stratified CV, reference rows = 128 random rows of each test fold, seed 42.
Datasets: iris, wine, breast_cancer, and two synthetic binary sets (`synthetic_8f`, `synthetic_40f`) that differ only in
dimensionality.

## Output files (`results/sweep/`)

| File | Content |
|---|---|
| `fold_results.csv` | **Complete raw data**: one row per (dataset, configuration, CV fold) = 8000 rows: accuracy, raw structural statistics, components, USII, simulation cost |
| `config_results.csv` | One row per (dataset, configuration): mean/std/SE accuracy, mean USII and components, simulation cost, Pareto flags |
| `threshold_summary.csv` | Headline threshold results (primary rule and budget) per dataset and family; `ALL` = pooled over datasets |
| `threshold_by_budget.csv` | Same for every accuracy rule x budget |
| `weight_sensitivity.csv`, `weight_sensitivity_dirichlet.csv` | Equal weights, DPC 0.15/0.45, drop-one-component ablations; 200 random weightings |
| `constants_sensitivity.csv` | max_depth in {6, 8, 10} x max_trees in {100, 300, 1000} |
| `run_metadata.json` | Seed, CV scheme, constants, library versions |
| `figures/` | `usii_vs_accuracy`, `usii_vs_simulation_cost`, `accuracy_vs_simulation_cost`, `threshold_summary`, `weight_sensitivity` |

`results/demo/<dataset>/` holds a single-split run per dataset: `model_results.csv` and `explanations_and_traces.json`
(bounded structure summaries and one exact prediction trace per model).

## Limitations

* USII is a structural proxy. It is **not** validated against human judgements; only its agreement with a separate
  cognitive-cost criterion (SC) is measured, and SC shares structure (path length) with DPC, so that agreement is partly by construction.
* Single decision trees always satisfy SC <= 9 here (depth <= 8), so no USII cut-off exists for them at B = 9; use the
  B = 5 and 7 rows of `threshold_by_budget.csv` for trees. Within a single tree, USII ranks depth the same way under almost any weights.
* The budget B and the accuracy tolerance are choices, not facts; conclusions should be read as a function of them.
* Standard errors come from repeated CV folds, which are not independent, so they are optimistic; the fixed-tolerance rules are provided for that reason.
* Feature use counts encoded columns; USII ignores feature semantics and user expertise.


## AI statement

 I created this repository with Claude Code (Claude Opus 5.5). After clarifying the scope
with it, it generated the code, tests and README based on my own projects as style references. I
reviewed and edited the result and checked that it runs. I take full responsibility for and
ownership of the content of this repository.

