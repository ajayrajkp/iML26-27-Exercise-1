"""End-to-end sweep: fit, aggregate, threshold analysis, sensitivity, figures, metadata."""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from . import metric
from .datasets import DATASETS
from .plots import make_all
from .sensitivity import constants_sensitivity, weight_sensitivity
from .sweep import GB_LEARNING_RATE, REF_SIZE, aggregate, build_grid, run_dataset
from .threshold import PRIMARY_BUDGET, PRIMARY_RULE, add_pareto, primary_summary, threshold_table


def analyze(fold_df, out):
    """Everything downstream of the per-fold results (also usable on a saved fold_results.csv)."""
    out = Path(out)
    cfg = add_pareto(aggregate(fold_df))
    cfg.to_csv(out / "config_results.csv", index=False)
    table = threshold_table(cfg)
    table.to_csv(out / "threshold_by_budget.csv", index=False)
    primary_summary(table).to_csv(out / "threshold_summary.csv", index=False)
    named, wsum = weight_sensitivity(fold_df, cfg)
    named.to_csv(out / "weight_sensitivity.csv", index=False)
    wsum.to_csv(out / "weight_sensitivity_dirichlet.csv", index=False)
    constants_sensitivity(fold_df, cfg).to_csv(out / "constants_sensitivity.csv", index=False)
    make_all(cfg, table, wsum, out / "figures")
    return cfg, table


def run(output="results/sweep", datasets=None, n_splits=5, n_repeats=5, seed=42):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    names = list(DATASETS) if datasets is None else list(datasets)
    parts = out / "parts"; parts.mkdir(exist_ok=True)
    frames = []
    for name in names:
        cache = parts / f"{name}_s{n_splits}x{n_repeats}_seed{seed}.csv"
        if cache.exists():  # resume: a finished dataset is never refitted
            print(f"Dataset {name}: using cached {cache.name}", flush=True)
            frames.append(pd.read_csv(cache))
            continue
        print(f"Dataset {name}", flush=True)
        df = run_dataset(name, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
        df.to_csv(cache, index=False)
        frames.append(df)
    fold_df = pd.concat(frames, ignore_index=True)
    fold_df.to_csv(out / "fold_results.csv", index=False)
    cfg, table = analyze(fold_df, out)
    meta = {"seed": seed, "cv": f"RepeatedStratifiedKFold(n_splits={n_splits}, n_repeats={n_repeats})",
            "datasets": names, "n_configs_per_dataset": len(build_grid()), "reference_rows_per_fold": REF_SIZE,
            "gb_learning_rate": GB_LEARNING_RATE, "weights": metric.WEIGHTS, "max_depth": metric.MAX_DEPTH,
            "max_trees": metric.MAX_TREES, "primary_budget": PRIMARY_BUDGET, "primary_accuracy_rule": PRIMARY_RULE,
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "scipy": scipy.__version__}
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    return fold_df, cfg, table
