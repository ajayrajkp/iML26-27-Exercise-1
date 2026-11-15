"""Run the Simulatability Index experiments."""

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .datasets import DATASETS
from .metric import LENIENT_LIMITS, MODERATE_LIMITS, calculate_index
from .plots import create_plots


SEEDS = (13, 42, 97)


def create_models(seed):
    """Create all model configurations for one random seed."""
    models = []

    for depth in (1, 2, 3, 4, 5, 6, 8):
        models.append(("Decision Tree", f"depth={depth}", DecisionTreeClassifier(max_depth=depth, random_state=seed)))

    for trees in (1, 5, 10, 25, 50):
        for depth in (2, 4, 6, 8):
            model = RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=seed, n_jobs=1)
            models.append(("Random Forest", f"trees={trees},depth={depth}", model))

    for stages in (1, 5, 10, 25, 50):
        for depth in (1, 2, 3):
            model = GradientBoostingClassifier(n_estimators=stages, max_depth=depth, random_state=seed)
            models.append(("Gradient Boosting", f"stages={stages},depth={depth}", model))

    return models


def run_models(limits):
    """Fit every model on every dataset and calculate the index."""
    rows = []
    for dataset_name, load_data in DATASETS.items():
        dataset = load_data()
        for seed in SEEDS:
            train_x, test_x, train_y, test_y = train_test_split(
                dataset.data, dataset.target, test_size=0.25, stratify=dataset.target, random_state=seed
            )
            reference_data = test_x[:min(128, len(test_x))]
            for family, settings, model in create_models(seed):
                model.fit(train_x, train_y)
                row = {
                    "dataset": dataset_name,
                    "seed": seed,
                    "model_family": family,
                    "model_settings": settings,
                    "accuracy": model.score(test_x, test_y),
                }
                row.update(calculate_index(model, reference_data, limits))
                rows.append(row)
    return pd.DataFrame(rows)


def create_recommendations(results):
    """Find the most accurate model and the best model above the threshold."""
    rows = []
    for (dataset, family), group in results.groupby(["dataset", "model_family"]):
        summary = group.groupby("model_settings", as_index=False).agg(
            mean_accuracy=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            minimum_index=("simulatability_index", "min"),
            mean_index=("simulatability_index", "mean"),
        )
        best = summary.sort_values(["mean_accuracy", "mean_index"], ascending=False).iloc[0]
        passing = summary[summary["minimum_index"] >= 50.0]
        best_passing = passing.sort_values(["mean_accuracy", "mean_index"], ascending=False).iloc[0]
        rows.append({
            "dataset": dataset,
            "model_family": family,
            "best_model": best["model_settings"],
            "best_model_accuracy": best["mean_accuracy"],
            "best_model_accuracy_sd": best["accuracy_sd"],
            "best_model_minimum_index": best["minimum_index"],
            "best_passing_model": best_passing["model_settings"],
            "best_passing_accuracy": best_passing["mean_accuracy"],
            "best_passing_accuracy_sd": best_passing["accuracy_sd"],
            "best_passing_minimum_index": best_passing["minimum_index"],
        })
    return pd.DataFrame(rows)


def create_sensitivity_results(results):
    """Recalculate the index for 27 combinations of workload limits."""
    rows = []
    for step_limit in (10.0, 15.0, 30.0):
        for rule_limit in (20.0, 30.0, 100.0):
            for feature_limit in (7, 10, 12):
                loads = pd.concat([
                    results["prediction_steps"] / step_limit,
                    results["total_rules"] / rule_limit,
                    results["used_features"] / feature_limit,
                ], axis=1)
                scores = 100.0 / (1.0 + loads.max(axis=1))
                for dataset in sorted(results["dataset"].unique()):
                    mask = results["dataset"] == dataset
                    dataset_scores = scores[mask]
                    rows.append({
                        "step_limit": step_limit,
                        "rule_limit": rule_limit,
                        "feature_limit": feature_limit,
                        "dataset": dataset,
                        "number_of_models": int(mask.sum()),
                        "number_passing": int((dataset_scores >= 50).sum()),
                        "fraction_passing": float((dataset_scores >= 50).mean()),
                    })
    return pd.DataFrame(rows)


def run_experiment(output="results"):
    """Run both protocols and save all experiment outputs."""
    output_folder = Path(output)
    output_folder.mkdir(parents=True, exist_ok=True)
    moderate_results = run_models(MODERATE_LIMITS)
    lenient_results = run_models(LENIENT_LIMITS)
    recommendations = create_recommendations(moderate_results)
    sensitivity = create_sensitivity_results(moderate_results)

    moderate_results.to_csv(output_folder / "moderate_results.csv", index=False)
    lenient_results.to_csv(output_folder / "lenient_results.csv", index=False)
    recommendations.to_csv(output_folder / "recommendations.csv", index=False)
    sensitivity.to_csv(output_folder / "sensitivity.csv", index=False)

    summary = {
        "metric": "Simulatability Index",
        "formula": "100 / (1 + max(prediction_steps/step_limit, total_rules/rule_limit, used_features/feature_limit))",
        "threshold": 50.0,
        "moderate_limits": MODERATE_LIMITS,
        "lenient_limits": LENIENT_LIMITS,
        "number_of_models": int(len(moderate_results)),
        "moderate_models_passing": int(moderate_results["passes_threshold"].sum()),
        "lenient_models_passing": int(lenient_results["passes_threshold"].sum()),
        "datasets": sorted(moderate_results["dataset"].unique().tolist()),
        "model_families": sorted(moderate_results["model_family"].unique().tolist()),
        "seeds": list(SEEDS),
        "scope": "Structural proxy based on declared workload limits; not a universal human threshold",
    }
    with (output_folder / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    create_plots(moderate_results, output_folder / "figures")
    return summary

