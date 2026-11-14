import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from ossi.ossi import calculate_ossi

def test_stump_load_and_score_match_formula():
    X = np.array([[0.0], [1.0]])
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, [0, 1])
    result = calculate_ossi(model, X)
    assert result.prediction_steps == pytest.approx(1.0)
    assert result.global_rules == 1
    assert result.features_used == 1
    assert result.moderate_load == pytest.approx(0.1)
    assert result.ossi_moderate == pytest.approx(100 / 1.1)
    assert result.moderate_pass is True

def test_budget_boundary_is_score_50_and_passes():
    X = np.array([[0.0], [1.0]])
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, [0, 1])
    budgets = {"prediction_steps": 1.0, "global_rules": 1.0, "features": 1.0}
    result = calculate_ossi(model, X, moderate_budgets=budgets)
    assert result.ossi_moderate == pytest.approx(50.0)
    assert result.moderate_pass is True

def test_multiclass_boosting_counts_all_class_trees():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(90, 3)); y = np.tile([0, 1, 2], 30)
    model = GradientBoostingClassifier(n_estimators=4, max_depth=1, random_state=0).fit(X, y)
    result = calculate_ossi(model, X[:10])
    assert result.n_trees == 12
    assert result.prediction_steps == pytest.approx(result.n_trees * result.mean_path_depth)
    assert result.global_rules == pytest.approx(result.n_trees * result.mean_internal_nodes_per_tree)

def test_rejects_nonpositive_budget():
    X = np.array([[0.0], [1.0]])
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, [0, 1])
    with pytest.raises(ValueError):
        calculate_ossi(model, X, moderate_budgets={"prediction_steps": 0, "global_rules": 1, "features": 1})

def test_random_forest_aggregates_total_rules_and_paths():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(80, 4)); y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=7, max_depth=3, random_state=5).fit(X, y)
    result = calculate_ossi(model, X[:15])
    assert result.n_trees == 7
    assert result.prediction_steps == pytest.approx(7 * result.mean_path_depth)
    assert result.global_rules == result.n_trees * result.mean_internal_nodes_per_tree

def test_oversized_workload_scores_below_fifty():
    X = np.array([[0.0], [1.0]])
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, [0, 1])
    budgets = {"prediction_steps": .5, "global_rules": .5, "features": .5}
    result = calculate_ossi(model, X, moderate_budgets=budgets)
    assert result.ossi_moderate < 50
    assert result.moderate_pass is False

def test_rejects_empty_reference_and_feature_mismatch():
    X = np.array([[0.0], [1.0]])
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, [0, 1])
    with pytest.raises(ValueError): calculate_ossi(model, np.empty((0, 1)))
    with pytest.raises(ValueError): calculate_ossi(model, np.ones((2, 2)))

def test_unfitted_or_unsupported_model_is_rejected():
    from sklearn.linear_model import LogisticRegression
    with pytest.raises(TypeError): calculate_ossi(LogisticRegression(), np.ones((2, 1)))

def test_score_always_in_range_and_load_is_max_component():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(100, 4)); y = (X[:, 0] > 0).astype(int)
    model = DecisionTreeClassifier(max_depth=4, random_state=3).fit(X, y)
    result = calculate_ossi(model, X[:20])
    assert 0 < result.ossi_moderate <= 100
    assert result.moderate_load == pytest.approx(max(result.prediction_steps / 15, result.global_rules / 30, result.features_used / 10))

def test_repeated_experiment_writes_per_run_summary_and_figures(tmp_path):
    from ossi.experiment import run
    per_run, summary = run(tmp_path, dataset="iris", seeds=(10, 11))
    assert len(per_run) == 6
    assert len(summary) == 3
    assert (tmp_path / "model_results.csv").exists()
    assert (tmp_path / "summary_results.csv").exists()
    assert (tmp_path / "ossi_analysis.csv").exists()
    assert (tmp_path / "ossi_protocol_comparison.png").exists()
    assert (tmp_path / "ossi_vs_balanced_accuracy.png").exists()
    assert set(per_run["seed"]) == {10, 11}

def test_one_hot_columns_count_as_original_feature_group():
    X = np.eye(3, dtype=float)
    y = np.array([0, 1, 2])
    model = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    result = calculate_ossi(model, X, feature_groups=["bean_type"] * 3)
    assert result.features_used == 1

def test_bank_loader_has_expected_target_and_excludes_duration():
    from ossi.experiment import load_bank_marketing
    data = load_bank_marketing()
    assert len(data.target) == 45211
    assert set(data.target.unique()) == {"yes", "no"}
    assert "duration" not in data.feature_names
    assert len(data.feature_names) == 15

def test_bank_encoding_fits_categories_on_training_partition():
    import pandas as pd
    from ossi.experiment import _encode_split
    train = pd.DataFrame({"job": ["admin", "blue"], "age": [20, 50]})
    test = pd.DataFrame({"job": ["unseen"], "age": [35]})
    ref = test.copy()
    X_train, X_test, X_ref, names, groups = _encode_split(train, test, ref, list(train.columns))
    assert X_train.shape[1] == 3
    assert X_test.shape == (1, 3)
    assert groups.count("job") == 2
    assert names[-1] == "age"
    assert np.all(X_test[0, :2] == 0)

def test_bank_marketing_experiment_writes_attributed_results(tmp_path):
    from ossi.experiment import run
    per_run, summary = run(tmp_path, dataset="bank_marketing", seeds=(42,))
    assert len(per_run) == 3
    assert len(summary) == 3
    assert {"balanced_accuracy", "macro_f1", "yes_precision", "yes_recall", "yes_f1"}.issubset(per_run.columns)
    assert {"yes_precision_mean", "yes_recall_mean", "yes_f1_mean"}.issubset(summary.columns)
    assert per_run["features_used"].max() <= 15
    assert per_run["encoded_feature_count"].min() > per_run["original_feature_count"].min()
    manifest = __import__("json").loads((tmp_path / "dataset_manifest.json").read_text())
    assert manifest["excluded_features"] == ["duration"]
    assert manifest["license"].startswith("CC BY 4.0")
