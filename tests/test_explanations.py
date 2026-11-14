import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from usii.explanations import model_explanation, prediction_trace


@pytest.fixture
def fitted():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3))
    y = (X[:, 0] > 0).astype(int)
    names = ["a", "b", "c"]
    return X, y, names


def test_trace_follows_exact_conditions_and_matches_prediction(fitted):
    X, y, names = fitted
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    trace = prediction_trace(model, X[0], names)
    assert trace["prediction"] == model.predict(X[:1])[0]
    for step in trace["paths"][0]["steps"]:
        holds = step["value"] <= step["threshold"] if step["operator"] == "<=" else step["value"] > step["threshold"]
        assert holds


def test_explanation_is_bounded_and_json_safe(fitted):
    import json
    X, y, names = fitted
    model = RandomForestClassifier(n_estimators=3, max_depth=6, random_state=0).fit(X, y)
    exp = model_explanation(model, names, name="RF", max_nodes=5)
    assert exp["tree_count"] == 3
    assert all(len(t["nodes"]) <= 5 for t in exp["trees"])
    json.dumps(exp)


def test_trace_rejects_wrong_row_length(fitted):
    X, y, names = fitted
    model = DecisionTreeClassifier(max_depth=2).fit(X, y)
    with pytest.raises(ValueError):
        prediction_trace(model, X[0][:2], names)
