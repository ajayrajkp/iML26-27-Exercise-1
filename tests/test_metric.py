import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from simulatability_index.datasets import load_banknote_data
from simulatability_index.metric import LENIENT_LIMITS, MODERATE_LIMITS, calculate_index, get_trees


def make_test_data():
    random = np.random.default_rng(7)
    features = random.normal(size=(100, 6))
    target = (features[:, 0] + features[:, 1] > 0).astype(int)
    return features, target


def test_score_is_repeatable():
    features, target = make_test_data()
    model = DecisionTreeClassifier(max_depth=3, random_state=1)
    model.fit(features, target)
    first = calculate_index(model, features[:30])
    second = calculate_index(model, features[:30])
    assert 0 < first["simulatability_index"] <= 100
    assert first == second


def test_threshold_matches_limits():
    features, target = make_test_data()
    model = RandomForestClassifier(n_estimators=25, max_depth=3, random_state=1)
    model.fit(features, target)

    for limits in (MODERATE_LIMITS, LENIENT_LIMITS):
        result = calculate_index(model, features[:30], limits)
        direct_check = (
            result["prediction_steps"] <= limits["steps"]
            and result["total_rules"] <= limits["rules"]
            and result["used_features"] <= limits["features"]
        )
        assert result["passes_threshold"] == direct_check


def test_lenient_score_is_not_lower():
    features, target = make_test_data()
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=1)
    model.fit(features, target)
    moderate = calculate_index(model, features[:30], MODERATE_LIMITS)
    lenient = calculate_index(model, features[:30], LENIENT_LIMITS)
    assert lenient["simulatability_index"] >= moderate["simulatability_index"]


def test_all_three_model_families_are_supported():
    features, target = make_test_data()
    models = [
        DecisionTreeClassifier(max_depth=3, random_state=1),
        RandomForestClassifier(n_estimators=5, max_depth=3, random_state=1),
        GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=1),
    ]

    for model in models:
        model.fit(features, target)
        result = calculate_index(model, features[:30])
        assert result["number_of_trees"] == len(get_trees(model))
        assert 0 < result["simulatability_index"] <= 100


def test_banknote_dataset():
    dataset = load_banknote_data()
    assert dataset.data.shape == (1372, 4)
    assert dataset.target.shape == (1372,)
    assert set(np.unique(dataset.target)) == {0, 1}

