"""Stable model summaries and exact prediction traces for sklearn tree models."""
import numpy as np
from .metric import extract_trees as _trees

def model_explanation(model, feature_names, name=None, max_nodes=31):
    """Produce a bounded JSON-safe tree summary."""
    summaries = []
    for est in _trees(model):
        tr = est.tree_; queue = [0]; items = []
        while queue and len(items) < max_nodes:
            node = queue.pop(0); left, right = int(tr.children_left[node]), int(tr.children_right[node])
            if left == right:
                items.append({"node": node, "leaf": True, "samples": int(tr.n_node_samples[node]), "value": np.asarray(tr.value[node]).tolist()})
            else:
                f = int(tr.feature[node])
                items.append({"node": node, "leaf": False, "feature": str(feature_names[f]), "threshold": float(tr.threshold[node]), "left": left, "right": right, "samples": int(tr.n_node_samples[node])})
                queue.extend((left, right))
        summaries.append({"nodes": items, "truncated": bool(queue), "node_count": int(tr.node_count), "depth": int(tr.max_depth)})
    return {"model": name or type(model).__name__, "tree_count": len(summaries), "class_names": [str(c) for c in getattr(model, "classes_", [])], "trees": summaries}

def prediction_trace(model, row, feature_names):
    """Return prediction plus the exact split conditions traversed in each tree."""
    x = np.asarray(row).reshape(1, -1)
    if x.shape[1] != len(feature_names): raise ValueError("row and feature_names must have the same length")
    paths = []
    for index, est in enumerate(_trees(model)):
        tr = est.tree_; node = 0; steps = []
        while tr.children_left[node] != tr.children_right[node]:
            f = int(tr.feature[node]); val = float(x[0, f]); threshold = float(tr.threshold[node]); left = val <= threshold
            steps.append({"feature": str(feature_names[f]), "value": val, "operator": "<=" if left else ">", "threshold": threshold})
            node = int(tr.children_left[node] if left else tr.children_right[node])
        paths.append({"tree": index, "steps": steps, "leaf": node, "leaf_value": np.asarray(tr.value[node]).tolist()})
    return {"prediction": np.asarray(model.predict(x)).reshape(-1)[0].item(), "paths": paths}
