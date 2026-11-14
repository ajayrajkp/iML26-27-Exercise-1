"""Repeated OSSI benchmarks with per-run results, summaries, and plots."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import Bunch
from .ossi import calculate_ossi
from .explanations import model_explanation, prediction_trace

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
BANK_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "bank-full.csv"

def load_bank_marketing(path=BANK_DATA_PATH):
    """Load UCI's 45,211-row bank-full.csv; exclude post-call duration to avoid leakage."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bank Marketing CSV not found at {path}; see data/README.md")
    frame = pd.read_csv(path, sep=";")
    if "y" not in frame.columns:
        raise ValueError("Bank Marketing CSV must contain target column 'y'")
    y = frame.pop("y")
    if "duration" in frame.columns:
        frame = frame.drop(columns=["duration"])
    return Bunch(data=frame, target=y, feature_names=list(frame.columns), target_names=["no", "yes"])

DATASETS = {"breast_cancer": load_breast_cancer, "iris": load_iris, "wine": load_wine, "bank_marketing": load_bank_marketing}

def _encode_split(X_train, X_test, X_reference, feature_names):
    """Fit categorical encoding on training rows and retain original feature groups."""
    if not isinstance(X_train, pd.DataFrame):
        names = [str(name) for name in feature_names]
        groups = list(names)
        return np.asarray(X_train), np.asarray(X_test), np.asarray(X_reference), names, groups
    categorical = list(X_train.select_dtypes(include=["object", "category", "bool"]).columns)
    numeric = [column for column in X_train.columns if column not in categorical]
    transformers = []
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), categorical))
    if numeric:
        transformers.append(("numeric", "passthrough", numeric))
    encoder = ColumnTransformer(transformers, remainder="drop")
    train_encoded = encoder.fit_transform(X_train).astype(np.float32)
    test_encoded = encoder.transform(X_test).astype(np.float32)
    ref_encoded = encoder.transform(X_reference).astype(np.float32)
    groups, names = [], []
    if categorical:
        fitted = encoder.named_transformers_["categorical"]
        for column, categories in zip(categorical, fitted.categories_):
            groups.extend([column] * len(categories))
            names.extend([f"{column}={category}" for category in categories])
    groups.extend(numeric); names.extend(numeric)
    return train_encoded, test_encoded, ref_encoded, names, groups

def _models(seed):
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=8, random_state=seed, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=seed),
    }

def run(output_dir="results", dataset="breast_cancer", seeds=DEFAULT_SEEDS, bank_data_path=None):
    """Run each model on a stratified split for every seed and aggregate results."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from: {', '.join(DATASETS)}")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain at least one unique integer")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    data = DATASETS[dataset](bank_data_path or BANK_DATA_PATH) if dataset == "bank_marketing" else DATASETS[dataset]()
    X, y = data.data, data.target
    rows, artifacts = [], {}
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=seed
        )
        X_reference_raw = X_test.iloc[:min(128, len(X_test))] if isinstance(X_test, pd.DataFrame) else X_test[:min(128, len(X_test))]
        X_train_model, X_test_model, X_reference, encoded_names, feature_groups = _encode_split(X_train, X_test, X_reference_raw, data.feature_names)
        baseline = DummyClassifier(strategy="most_frequent").fit(X_train_model, y_train)
        baseline_prediction = baseline.predict(X_test_model)
        for name, model in _models(seed).items():
            model.fit(X_train_model, y_train)
            predictions = model.predict(X_test_model)
            score = calculate_ossi(model, X_reference, feature_groups=feature_groups)
            row = {
                "dataset": dataset, "seed": seed, "model": name,
                "accuracy": float(model.score(X_test_model, y_test)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
                "majority_baseline_accuracy": float(baseline.score(X_test_model, y_test)),
                "majority_baseline_balanced_accuracy": float(balanced_accuracy_score(y_test, baseline_prediction)),
                "encoded_feature_count": int(X_train_model.shape[1]),
                "original_feature_count": int(len(data.feature_names)), **score.as_dict(),
            }
            if dataset == "bank_marketing":
                row.update({
                    "yes_precision": float(precision_score(y_test, predictions, pos_label="yes", zero_division=0)),
                    "yes_recall": float(recall_score(y_test, predictions, pos_label="yes", zero_division=0)),
                    "yes_f1": float(f1_score(y_test, predictions, pos_label="yes", zero_division=0)),
                })
            rows.append(row)
            # Keep inspectable examples without serializing every ensemble for every seed.
            if seed == seeds[0]:
                artifacts[name] = {
                    "seed": seed,
                    "model_explanation": model_explanation(model, encoded_names, name=name, max_nodes=31),
                    "prediction_trace": prediction_trace(model, X_test_model[0], encoded_names),
                }
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "model_results.csv", index=False)
    summary = per_run.groupby("model", sort=False).agg(
        runs=("seed", "count"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        majority_baseline_accuracy_mean=("majority_baseline_accuracy", "mean"),
        majority_baseline_balanced_accuracy_mean=("majority_baseline_balanced_accuracy", "mean"),
        ossi_moderate_mean=("ossi_moderate", "mean"), ossi_moderate_std=("ossi_moderate", "std"),
        moderate_pass_rate=("moderate_pass", "mean"),
        ossi_lenient_mean=("ossi_lenient", "mean"), ossi_lenient_std=("ossi_lenient", "std"),
        lenient_pass_rate=("lenient_pass", "mean"),
        prediction_steps_mean=("prediction_steps", "mean"), prediction_steps_std=("prediction_steps", "std"),
        global_rules_mean=("global_rules", "mean"), global_rules_std=("global_rules", "std"),
        features_used_mean=("features_used", "mean"), features_used_std=("features_used", "std"),
    ).reset_index()
    if dataset == "bank_marketing":
        positive_summary = per_run.groupby("model", sort=False).agg(
            yes_precision_mean=("yes_precision", "mean"), yes_precision_std=("yes_precision", "std"),
            yes_recall_mean=("yes_recall", "mean"), yes_recall_std=("yes_recall", "std"),
            yes_f1_mean=("yes_f1", "mean"), yes_f1_std=("yes_f1", "std"),
        ).reset_index()
        summary = summary.merge(positive_summary, on="model", validate="one_to_one")
    summary.insert(0, "dataset", dataset)
    summary.to_csv(out / "summary_results.csv", index=False)
    analysis_columns = ["dataset", "seed", "model", "accuracy", "balanced_accuracy", "macro_f1", "majority_baseline_accuracy", "majority_baseline_balanced_accuracy", "encoded_feature_count", "original_feature_count", "prediction_steps", "global_rules", "features_used", "moderate_load", "ossi_moderate", "moderate_pass", "lenient_load", "ossi_lenient", "lenient_pass"]
    analysis_columns.extend(column for column in ("yes_precision", "yes_recall", "yes_f1") if column in per_run.columns)
    per_run[analysis_columns].to_csv(out / "ossi_analysis.csv", index=False)
    with (out / "explanations_and_traces.json").open("w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)
    manifest = {"dataset": dataset, "seeds": list(seeds), "reference_rows_per_seed": 128, "target": "y" if dataset == "bank_marketing" else "dataset target", "categorical_features_grouped_for_ossi": dataset == "bank_marketing"}
    if dataset == "bank_marketing":
        manifest.update({
            "source": "UCI Machine Learning Repository, Bank Marketing (bank-full.csv)",
            "source_url": "https://archive.ics.uci.edu/dataset/222/bank",
            "citation": "Moro, Rita, and Cortez (2014), DOI: 10.24432/C5K306",
            "license": "CC BY 4.0; attribution required",
            "file_sha256": hashlib.sha256(Path(bank_data_path or BANK_DATA_PATH).read_bytes()).hexdigest(),
            "excluded_features": ["duration"],
            "exclusion_reason": "Duration is only known after the last call and can leak post-outcome information into prediction.",
        })
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary)); width = 0.36
    ax.bar(x - width / 2, summary.ossi_moderate_mean, width, yerr=summary.ossi_moderate_std.fillna(0), capsize=4, label="Moderate")
    ax.bar(x + width / 2, summary.ossi_lenient_mean, width, yerr=summary.ossi_lenient_std.fillna(0), capsize=4, label="Lenient")
    ax.axhline(50, color="#b44", linestyle="--", label="Budget boundary (50)")
    ax.set_xticks(x, summary.model); ax.set_ylim(0, 100)
    ax.set(ylabel="OSSI (0–100)", title=f"OSSI protocol comparison — {dataset} ({len(seeds)} seeds)")
    ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out / "ossi_protocol_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, group in per_run.groupby("model", sort=False):
        ax.scatter(group.ossi_moderate, group.balanced_accuracy, label=name, alpha=.8)
    ax.axvline(50, color="#b44", linestyle="--", label="Moderate budget boundary")
    ax.set(xlabel="Moderate OSSI (0–100)", ylabel="Balanced accuracy", title=f"OSSI and balanced accuracy across seeds — {dataset}")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / "ossi_vs_balanced_accuracy.png", dpi=160); plt.close(fig)
    return per_run, summary

if __name__ == "__main__":
    _, result = run()
    print(result.to_string(index=False))
