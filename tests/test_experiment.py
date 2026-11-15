from simulatability_index.datasets import DATASETS
from simulatability_index.experiment import create_models


def test_experiment_size_and_model_families():
    models = create_models(seed=42)
    families = {family for family, settings, model in models}

    assert len(DATASETS) == 4
    assert len(models) == 42
    assert families == {"Decision Tree", "Random Forest", "Gradient Boosting"}

