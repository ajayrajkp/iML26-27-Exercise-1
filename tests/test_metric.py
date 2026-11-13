import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from usii.metric import MAX_DEPTH, MAX_TREES, WEIGHTS, calculate_usii, extract_trees, structure_stats


@pytest.fixture
def data():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 5))
    return X, (X[:, 0] + X[:, 1] > 0).astype(int)


FACTORIES = [lambda: DecisionTreeClassifier(max_depth=3, random_state=1),
             lambda: RandomForestClassifier(n_estimators=5, max_depth=3, random_state=1),
             lambda: GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=1)]


@pytest.mark.parametrize("factory", FACTORIES)
def test_components_bounded_and_reproducible(data, factory):
    X, y = data
    model = factory().fit(X, y)
    a, b = calculate_usii(model, X[:30]), calculate_usii(model, X[:30])
    assert 0 <= a.usii <= 100
    assert all(0 <= v <= 1 for v in (a.dpc, a.bc, a.fd, a.ec))
    assert a == b


def test_known_value_single_stump():
    # One split on one feature: every row has path length 1, one internal node, one feature used.
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, [0, 0, 1, 1])
    r = calculate_usii(model, X)
    assert (r.n_trees, r.mean_path_depth, r.mean_internal_nodes, r.n_used_features) == (1, 1.0, 1.0, 1)
    assert r.sim_cost == 1.0
    expected = 100 * (1 - (0.30 * 1 / MAX_DEPTH + 0.30 * 1 / (2 ** MAX_DEPTH - 1) + 0.20 * 1 + 0.20 * 1 / MAX_TREES))
    assert r.usii == pytest.approx(expected)


def test_simulation_cost_is_sum_over_trees(data):
    X, y = data
    model = RandomForestClassifier(n_estimators=7, max_depth=3, random_state=0).fit(X, y)
    s = structure_stats(model, X[:40])
    assert s["sim_cost"] == pytest.approx(s["mean_path_depth"] * s["n_trees"])


def test_more_structure_never_raises_usii(data):
    X, y = data
    shallow = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    deep = DecisionTreeClassifier(max_depth=8, random_state=0).fit(X, y)
    assert calculate_usii(shallow, X[:50]).usii >= calculate_usii(deep, X[:50]).usii


def test_multiclass_boosting_counts_one_tree_per_class_per_stage():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 4))
    y = np.digitize(X[:, 0], [-0.5, 0.5])
    model = GradientBoostingClassifier(n_estimators=4, max_depth=2, random_state=0).fit(X, y)
    assert len(extract_trees(model)) == 4 * 3


def test_reference_data_changes_path_component(data):
    X, y = data
    model = DecisionTreeClassifier(max_depth=4, random_state=1).fit(X, y)
    assert calculate_usii(model, X[:5]).mean_path_depth != calculate_usii(model, X).mean_path_depth


def test_rejects_empty_reference_and_bad_inputs(data):
    X, y = data
    model = DecisionTreeClassifier().fit(X, y)
    with pytest.raises(ValueError):
        calculate_usii(model, np.empty((0, X.shape[1])))
    with pytest.raises(ValueError):
        calculate_usii(model, X[:, :3])
    with pytest.raises(TypeError):
        extract_trees(object())


@pytest.mark.parametrize("bad", [{"DPC": 0.5, "BC": 0.5, "FD": 0.5, "EC": 0.5},
                                 {"DPC": 1.2, "BC": -0.2, "FD": 0.0, "EC": 0.0},
                                 {"DPC": 1.0}])
def test_rejects_bad_weights(data, bad):
    X, y = data
    model = DecisionTreeClassifier(max_depth=2).fit(X, y)
    with pytest.raises(ValueError):
        calculate_usii(model, X[:10], weights=bad)


def test_custom_weights_match_manual_formula(data):
    X, y = data
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    w = {"DPC": 0.5, "BC": 0.1, "FD": 0.3, "EC": 0.1}
    r = calculate_usii(model, X[:30], weights=w)
    manual = 100 * (1 - (w["DPC"] * r.dpc + w["BC"] * r.bc + w["FD"] * r.fd + w["EC"] * r.ec))
    assert r.usii == pytest.approx(manual)
    assert set(WEIGHTS) == set(w)
