"""Structural statistics and the USII score for sklearn tree-based classifiers.

USII (Unified Structural Interpretability Index) is defined *within* a model class
(single tree, random forest, gradient-boosted trees): it ranks models of the same
family by how much structure a person would have to take in.

    USII = 100 * (1 - (w_DPC*DPC + w_BC*BC + w_FD*FD + w_EC*EC))

All components are in [0, 1] (clipped). Higher USII = less measured structure.
Raw statistics are kept separate from the score so that weights and normalisation
constants can be changed afterwards without refitting any model.

A separate, non-normalised quantity, the *simulation cost* (SC), is also reported:
the number of split conditions a person must evaluate to reproduce one prediction
by hand (sum of root-to-leaf path lengths over all trees). SC is the cognitive-load
measure used for the threshold analysis; USII is the index being tested against it.
"""
from dataclasses import asdict, dataclass

import numpy as np

WEIGHTS = {"DPC": 0.30, "BC": 0.30, "FD": 0.20, "EC": 0.20}
COMPONENTS = ("DPC", "BC", "FD", "EC")
MAX_DEPTH = 8     # depth budget (DPC) and tree-capacity depth (BC)
MAX_TREES = 300   # tree budget (EC). 300 = 100 boosting stages x 3 classes, so the
                  # sweeps in this project never saturate (the v0 prototype used 100).


@dataclass(frozen=True)
class USIIResult:
    dpc: float
    bc: float
    fd: float
    ec: float
    usii: float
    n_trees: int
    mean_path_depth: float
    mean_internal_nodes: float
    total_internal_nodes: int
    n_used_features: int
    n_features: int
    sim_cost: float

    def as_dict(self):
        return asdict(self)


def extract_trees(model):
    """Return fitted sklearn trees from a DecisionTree, RandomForest or GradientBoosting model."""
    if hasattr(model, "tree_"):
        return [model]
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        raise TypeError("Expected a fitted DecisionTree, RandomForest, or GradientBoosting classifier.")
    trees = [est for est in np.asarray(estimators, dtype=object).ravel() if hasattr(est, "tree_")]
    if not trees:
        raise TypeError("No fitted decision trees found.")
    return trees


def structure_stats(model, X_reference):
    """Raw structural statistics of a fitted tree model on fixed reference rows."""
    X = np.asarray(X_reference)
    if X.ndim != 2 or len(X) == 0 or X.shape[1] == 0:
        raise ValueError("X_reference must be a non-empty 2D array")
    trees = extract_trees(model)
    per_row_total = np.zeros(len(X))
    nodes, used = [], set()
    for est in trees:
        tr = est.tree_
        if tr.n_features != X.shape[1]:
            raise ValueError("X_reference feature count does not match fitted model")
        edges = np.asarray(est.decision_path(X).sum(axis=1)).ravel().astype(float) - 1.0
        per_row_total += edges
        nodes.append(float(tr.node_count - tr.n_leaves))
        used.update(int(i) for i in tr.feature[tr.feature >= 0])
    return {
        "n_trees": len(trees),
        "mean_path_depth": float(per_row_total.mean() / len(trees)),
        "mean_internal_nodes": float(np.mean(nodes)),
        "total_internal_nodes": int(sum(nodes)),
        "n_used_features": len(used),
        "n_features": int(X.shape[1]),
        "sim_cost": float(per_row_total.mean()),
    }


def components_from_stats(mean_path_depth, mean_internal_nodes, n_used_features, n_features,
                          n_trees, max_depth=MAX_DEPTH, max_trees=MAX_TREES):
    """Normalised, clipped components. Works on scalars or numpy arrays."""
    if max_depth < 1 or max_trees < 1:
        raise ValueError("max_depth and max_trees must be positive")
    capacity = (2 ** max_depth) - 1
    return {
        "DPC": np.clip(np.asarray(mean_path_depth, dtype=float) / max_depth, 0, 1),
        "BC": np.clip(np.asarray(mean_internal_nodes, dtype=float) / capacity, 0, 1),
        "FD": np.clip(np.asarray(n_used_features, dtype=float) / np.asarray(n_features, dtype=float), 0, 1),
        "EC": np.clip(np.asarray(n_trees, dtype=float) / max_trees, 0, 1),
    }


def check_weights(weights):
    if set(weights) != set(COMPONENTS) or not np.isclose(sum(weights.values()), 1.0) \
            or any(v < 0 for v in weights.values()):
        raise ValueError("weights must be nonnegative DPC/BC/FD/EC weights summing to 1")


def usii_from_components(components, weights=None):
    """USII in [0, 100] from component values (scalars or arrays)."""
    w = WEIGHTS if weights is None else weights
    check_weights(w)
    penalty = sum(w[k] * components[k] for k in COMPONENTS)
    return np.clip(100.0 * (1.0 - penalty), 0, 100)


def calculate_usii(model, X_reference, *, max_depth=MAX_DEPTH, max_trees=MAX_TREES, weights=None):
    """USII for a fitted tree model; reference rows should be fixed across compared models."""
    if weights is not None:
        check_weights(weights)
    s = structure_stats(model, X_reference)
    c = components_from_stats(s["mean_path_depth"], s["mean_internal_nodes"], s["n_used_features"],
                              s["n_features"], s["n_trees"], max_depth, max_trees)
    score = float(usii_from_components(c, weights))
    return USIIResult(float(c["DPC"]), float(c["BC"]), float(c["FD"]), float(c["EC"]), score, **s)
