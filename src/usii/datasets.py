"""Dataset loaders: three scikit-learn built-ins plus two controlled synthetic sets.

The synthetic sets differ only in dimensionality, which lets the threshold analysis
ask whether the interpretability threshold depends on the number of input features.
Their generation seed is fixed and independent of the cross-validation seed.
"""
from collections import namedtuple

from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_classification

Dataset = namedtuple("Dataset", "data target feature_names")


def _builtin(loader):
    def load():
        d = loader()
        return Dataset(d.data, d.target, [str(n) for n in d.feature_names])
    return load


def _synthetic(n_features, n_informative, n_redundant):
    def load():
        X, y = make_classification(n_samples=1000, n_features=n_features, n_informative=n_informative,
                                   n_redundant=n_redundant, n_classes=2, flip_y=0.02,
                                   class_sep=1.0, random_state=0)
        return Dataset(X, y, [f"x{i}" for i in range(n_features)])
    return load


DATASETS = {
    "iris": _builtin(load_iris),
    "wine": _builtin(load_wine),
    "breast_cancer": _builtin(load_breast_cancer),
    "synthetic_8f": _synthetic(8, 4, 2),
    "synthetic_40f": _synthetic(40, 8, 4),
}
