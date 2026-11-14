"""Utilities for extracting fitted sklearn tree classifiers."""
import numpy as np

def extract_trees(model):
    """Return fitted decision trees from a tree, forest, or boosting classifier."""
    if hasattr(model, "tree_"):
        return [model]
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        raise TypeError("Expected a fitted decision tree or tree ensemble classifier")
    trees = [est for est in np.asarray(estimators, dtype=object).ravel() if hasattr(est, "tree_")]
    if not trees:
        raise TypeError("No fitted decision trees found")
    return trees
