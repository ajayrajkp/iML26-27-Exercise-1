"""Single-split demonstration: three reference models, standardised summaries, prediction traces."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .datasets import DATASETS
from .explanations import model_explanation, prediction_trace
from .metric import MAX_DEPTH, calculate_usii
from .sweep import GB_LEARNING_RATE, REF_SIZE


def run(output_dir="results/demo", seed=42, dataset="breast_cancer"):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from: {', '.join(DATASETS)}")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    data = DATASETS[dataset]()
    X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.25, stratify=data.target, random_state=seed)
    ref = np.random.default_rng(seed).choice(len(X_te), size=min(REF_SIZE, len(X_te)), replace=False)
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=MAX_DEPTH, random_state=seed, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=GB_LEARNING_RATE, random_state=seed),
    }
    rows, artifacts = [], {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        m = calculate_usii(model, X_te[ref])
        rows.append({"model": name, "dataset": dataset, "seed": seed, "accuracy": model.score(X_te, y_te), **m.as_dict()})
        artifacts[name] = {"model_explanation": model_explanation(model, data.feature_names, name=name, max_nodes=31),
                           "prediction_trace": prediction_trace(model, X_te[0], data.feature_names)}
    df = pd.DataFrame(rows)
    df.to_csv(out / "model_results.csv", index=False)
    (out / "explanations_and_traces.json").write_text(json.dumps(artifacts, indent=2))
    return df
