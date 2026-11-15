"""Calculate the Simulatability Index for fitted tree-based models."""

import numpy as np


MODERATE_LIMITS = {"steps": 15.0, "rules": 30.0, "features": 10}
LENIENT_LIMITS = {"steps": 30.0, "rules": 100.0, "features": 10}


def get_trees(model):
    """Return the component trees from a fitted tree model."""
    if hasattr(model, "tree_"):
        return [model]

    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        raise TypeError("Expected a fitted decision tree, random forest, or gradient boosting model")

    trees = []
    for tree in np.asarray(estimators, dtype=object).ravel():
        if hasattr(tree, "tree_"):
            trees.append(tree)

    if not trees:
        raise TypeError("No fitted trees were found")
    return trees


def calculate_index(model, reference_data, limits=MODERATE_LIMITS):
    """Calculate the index and return the score and workload measurements."""
    data = np.asarray(reference_data)
    if data.ndim != 2 or len(data) == 0 or data.shape[1] == 0:
        raise ValueError("reference_data must be a non-empty two-dimensional array")

    for name in ("steps", "rules", "features"):
        if name not in limits or limits[name] <= 0:
            raise ValueError("limits must contain positive steps, rules, and features values")

    trees = get_trees(model)
    path_lengths = []
    total_rules = 0
    used_features = set()

    for tree in trees:
        tree_data = tree.tree_
        if tree_data.n_features != data.shape[1]:
            raise ValueError("The reference data does not match the fitted model")

        paths = tree.decision_path(data)
        lengths = np.asarray(paths.sum(axis=1)).ravel() - 1
        path_lengths.extend(lengths)
        total_rules += int(tree_data.node_count - tree_data.n_leaves)

        for feature in tree_data.feature:
            if feature >= 0:
                used_features.add(int(feature))

    average_path = float(np.mean(path_lengths))
    prediction_steps = float(len(trees) * average_path)
    feature_count = len(used_features)
    step_load = prediction_steps / limits["steps"]
    rule_load = total_rules / limits["rules"]
    feature_load = feature_count / limits["features"]
    largest_load = float(max(step_load, rule_load, feature_load))
    score = float(100.0 / (1.0 + largest_load))

    return {
        "simulatability_index": score,
        "passes_threshold": score >= 50.0,
        "prediction_steps": prediction_steps,
        "total_rules": total_rules,
        "used_features": feature_count,
        "step_load": float(step_load),
        "rule_load": float(rule_load),
        "feature_load": float(feature_load),
        "largest_load": largest_load,
        "number_of_trees": len(trees),
        "average_path_length": average_path,
    }

