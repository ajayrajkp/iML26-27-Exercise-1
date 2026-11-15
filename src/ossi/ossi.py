"""Operational Structural Simulatability Index for sklearn tree ensembles."""
from dataclasses import asdict, dataclass
import numpy as np
from .models import extract_trees

MODERATE_BUDGETS = {"prediction_steps": 15.0, "global_rules": 30.0, "features": 10.0}
LENIENT_BUDGETS = {"prediction_steps": 30.0, "global_rules": 100.0, "features": 10.0}


@dataclass(frozen=True)
class OSSIResult:
    prediction_steps: float
    global_rules: int
    features_used: int
    moderate_load: float
    ossi_moderate: float
    moderate_pass: bool
    lenient_load: float
    ossi_lenient: float
    lenient_pass: bool
    n_trees: int
    mean_path_depth: float
    mean_internal_nodes_per_tree: float

    def as_dict(self):
        return asdict(self)


def calculate_ossi(model, X_reference, *, feature_groups=None, moderate_budgets=None, lenient_budgets=None):
    """Calculate OSSI for a fixed set of representative rows.

    P is tree_count times mean root-to-leaf edge depth across trees and rows.
    This counts each constituent tree path, including class-specific boosting
    trees, as required by the declared manual simulation protocol. R sums all
    internal nodes; F counts distinct split features. The returned pass flags
    mean only that all three structural workloads fit the stated budgets.
    """
    X = np.asarray(X_reference)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X_reference must be a non-empty 2D array")
    if feature_groups is not None and len(feature_groups) != X.shape[1]:
        raise ValueError("feature_groups length must match X_reference columns")
    trees = extract_trees(model)
    all_depths, total_rules, used = [], 0, set()
    for tree in trees:
        tr = tree.tree_
        if tr.n_features != X.shape[1]:
            raise ValueError("X_reference feature count does not match fitted model")
        path_sizes = np.asarray(tree.decision_path(X).sum(axis=1)).ravel()
        all_depths.extend((path_sizes - 1).astype(float))
        total_rules += int(tr.node_count - tr.n_leaves)
        for index in tr.feature[tr.feature >= 0]:
            index = int(index)
            used.add(feature_groups[index] if feature_groups is not None else index)
    mean_depth = float(np.mean(all_depths))
    P = float(len(trees) * mean_depth)
    R = int(total_rules)
    F = len(used)

    def score(budgets):
        expected = {"prediction_steps", "global_rules", "features"}
        if set(budgets) != expected or any(not np.isfinite(v) or v <= 0 for v in budgets.values()):
            raise ValueError("each OSSI budget must be finite and positive")
        load = max(P / budgets["prediction_steps"], R / budgets["global_rules"], F / budgets["features"])
        return float(load), float(100.0 / (1.0 + load)), bool(load <= 1.0)

    moderate_load, moderate, moderate_pass = score(MODERATE_BUDGETS if moderate_budgets is None else moderate_budgets)
    lenient_load, lenient, lenient_pass = score(LENIENT_BUDGETS if lenient_budgets is None else lenient_budgets)
    return OSSIResult(P, R, F, moderate_load, moderate, moderate_pass, lenient_load, lenient, lenient_pass, len(trees), mean_depth, float(R / len(trees)))
