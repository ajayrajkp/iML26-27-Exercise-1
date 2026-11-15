"""Load the four datasets used in the experiment."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.utils import Bunch


def load_banknote_data():
    """Load the local UCI Banknote Authentication dataset."""
    project_folder = Path(__file__).resolve().parents[2]
    path = project_folder / "data" / "banknote_authentication.csv"
    columns = ["variance", "skewness", "curtosis", "entropy", "class"]
    frame = pd.read_csv(path, header=None, names=columns)

    if frame.shape != (1372, 5):
        raise ValueError("The banknote dataset has an unexpected size")
    if frame.isna().any().any():
        raise ValueError("The banknote dataset contains missing values")

    return Bunch(
        data=frame[columns[:-1]].to_numpy(dtype=float),
        target=frame["class"].to_numpy(dtype=int),
        feature_names=np.array(columns[:-1]),
    )


DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "banknote_authentication_uci": load_banknote_data,
}

