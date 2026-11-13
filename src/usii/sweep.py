"""Complexity sweeps: fit every model configuration under repeated stratified CV and
record accuracy together with the raw structural statistics and USII components."""
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from .datasets import DATASETS
from .metric import components_from_stats, structure_stats, usii_from_components

REF_SIZE = 128
GB_LEARNING_RATE = 0.3  # fixed so that small boosted ensembles can reach useful accuracy

DT_DEPTHS = tuple(range(1, 9))
RF_TREES = (1, 3, 5, 10, 25, 50, 100)
RF_DEPTHS = (2, 3, 4, 6, 8)
GB_STAGES = (1, 3, 5, 10, 25, 50, 100)
GB_DEPTHS = (1, 2, 3)


def build_grid():
    """List of (family, max_depth, n_estimators) configurations."""
    grid = [("Decision Tree", d, 1) for d in DT_DEPTHS]
    grid += [("Random Forest", d, n) for n, d in itertools.product(RF_TREES, RF_DEPTHS)]
    grid += [("Gradient Boosting", d, n) for n, d in itertools.product(GB_STAGES, GB_DEPTHS)]
    return grid


def config_id(family, depth, n_est):
    return f"{family}|depth={depth}|trees={n_est}"


def make_model(family, depth, n_est, seed):
    if family == "Decision Tree":
        return DecisionTreeClassifier(max_depth=depth, random_state=seed)
    if family == "Random Forest":
        return RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=seed, n_jobs=1)
    if family == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=n_est, max_depth=depth,
                                          learning_rate=GB_LEARNING_RATE, random_state=seed)
    raise ValueError(f"Unknown family {family!r}")


def run_dataset(name, n_splits=5, n_repeats=3, seed=42, grid=None, verbose=True):
    """Fit every configuration on every CV fold of one dataset. Returns a per-fold DataFrame."""
    ds = DATASETS[name]()
    X, y = ds.data, ds.target
    grid = build_grid() if grid is None else grid
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for fold_id, (tr, te) in enumerate(cv.split(X, y)):
        # Fixed reference rows for this fold, shared by every configuration.
        rng = np.random.default_rng(seed * 1000 + fold_id)
        ref = rng.choice(te, size=min(REF_SIZE, len(te)), replace=False)
        for family, depth, n_est in grid:
            model = make_model(family, depth, n_est, seed * 1000 + fold_id).fit(X[tr], y[tr])
            stats = structure_stats(model, X[ref])
            comp = components_from_stats(stats["mean_path_depth"], stats["mean_internal_nodes"],
                                         stats["n_used_features"], stats["n_features"], stats["n_trees"])
            rows.append({
                "dataset": name, "family": family, "config_id": config_id(family, depth, n_est),
                "max_depth": depth, "n_estimators": n_est, "repeat": fold_id // n_splits,
                "fold": fold_id % n_splits, "n_train": len(tr), "n_test": len(te),
                "accuracy": float(model.score(X[te], y[te])), **stats,
                "dpc": float(comp["DPC"]), "bc": float(comp["BC"]), "fd": float(comp["FD"]),
                "ec": float(comp["EC"]), "usii": float(usii_from_components(comp)),
            })
        if verbose:
            print(f"  {name}: fold {fold_id + 1}/{n_splits * n_repeats} done", flush=True)
    return pd.DataFrame(rows)


def aggregate(fold_df):
    """One row per (dataset, config): mean/std/SE of accuracy and mean of structural columns."""
    keys = ["dataset", "family", "config_id", "max_depth", "n_estimators"]
    g = fold_df.groupby(keys, sort=False)
    out = g.agg(n_folds=("accuracy", "size"), acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                usii=("usii", "mean"), usii_std=("usii", "std"), dpc=("dpc", "mean"), bc=("bc", "mean"),
                fd=("fd", "mean"), ec=("ec", "mean"), n_trees=("n_trees", "mean"),
                mean_path_depth=("mean_path_depth", "mean"), total_internal_nodes=("total_internal_nodes", "mean"),
                n_used_features=("n_used_features", "mean"), sim_cost=("sim_cost", "mean")).reset_index()
    out["acc_se"] = out["acc_std"] / np.sqrt(out["n_folds"])
    return out
