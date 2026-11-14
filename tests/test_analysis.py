import numpy as np
import pandas as pd
import pytest

from usii.sensitivity import dirichlet_schemes, named_weight_schemes
from usii.threshold import accurate_mask, add_pareto, pareto_flags, threshold_table, usii_cutoff


def toy_configs():
    return pd.DataFrame({
        "dataset": "d", "family": "F",
        "config_id": [f"c{i}" for i in range(5)],
        "acc_mean": [0.70, 0.90, 0.94, 0.95, 0.95],
        "acc_se": [0.01, 0.01, 0.01, 0.01, 0.01],
        "usii": [95, 85, 70, 55, 40],
        "sim_cost": [1, 3, 8, 30, 90],
    })


def test_accuracy_rules():
    df = toy_configs()
    assert accurate_mask(df, "1se").tolist() == [False, False, True, True, True]
    assert accurate_mask(df, "tol_0.05").tolist() == [False, True, True, True, True]


def test_usii_cutoff_separates_perfectly_ordered_data():
    auc, tau = usii_cutoff(np.array([95, 85, 70, 55, 40]), np.array([1, 1, 1, 0, 0]))
    assert auc == 1.0
    assert 55 < tau <= 70


def test_usii_cutoff_undefined_when_one_class():
    auc, tau = usii_cutoff(np.array([1.0, 2.0]), np.array([True, True]))
    assert np.isnan(auc) and np.isnan(tau)


def test_threshold_table_feasibility_and_knee():
    t = threshold_table(toy_configs(), budgets=(2, 9), rules=("1se",))
    row9 = t[(t.dataset == "d") & (t.budget == 9)].iloc[0]
    assert row9.feasible and row9.knee_config == "c2"      # most interpretable accurate model
    assert row9.best_feasible_config == "c2"
    row2 = t[(t.dataset == "d") & (t.budget == 2)].iloc[0]
    assert not row2.feasible                                # no accurate model costs <= 2 conditions


def test_pareto_flags():
    df = pd.DataFrame({"usii": [90, 80, 70, 75], "acc_mean": [0.6, 0.9, 0.95, 0.85]})
    assert pareto_flags(df).tolist() == [True, True, True, False]
    assert "pareto_in_family" in add_pareto(toy_configs()).columns


def test_weight_schemes_are_valid():
    for w in list(named_weight_schemes().values()) + list(dirichlet_schemes(20).values()):
        assert sum(w.values()) == pytest.approx(1.0)
        assert min(w.values()) >= 0
