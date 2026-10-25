"""
Downloads a mix of well-known clean datasets into data/ for demoing
generic_prep.py / generic_train.py on a variety of classification and
regression problems.

Run: python scripts/download_demo_datasets.py
"""
import pathlib

import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"

SEABORN_PENGUINS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
UCI_ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
UCI_WINE_QUALITY_RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def save(df: pd.DataFrame, name: str) -> None:
    path = DATA / name
    df.to_csv(path, index=False)
    print(f"{name:28s} {df.shape[0]:>6} rows x {df.shape[1]:>2} cols -> {path}")


# ---------------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------------

# Palmer Penguins -- already present from an earlier demo (data/penguins.csv,
# target=species). Left untouched here.

iris = load_iris(as_frame=True)
iris_df = iris.frame.copy()
iris_df["species"] = iris_df["target"].map(dict(enumerate(iris.target_names)))
iris_df = iris_df.drop(columns=["target"])
save(iris_df, "iris.csv")

wine = load_wine(as_frame=True)
wine_df = wine.frame.copy()
wine_df["wine_class"] = wine_df["target"].map(dict(enumerate(wine.target_names)))
wine_df = wine_df.drop(columns=["target"])
save(wine_df, "wine.csv")

cancer = load_breast_cancer(as_frame=True)
cancer_df = cancer.frame.copy()
# sklearn encodes 0=malignant, 1=benign; keep the readable label for the demo.
cancer_df["diagnosis"] = cancer_df["target"].map(dict(enumerate(cancer.target_names)))
cancer_df = cancer_df.drop(columns=["target"])
save(cancer_df, "breast_cancer.csv")

_adult_cols = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]
adult_df = pd.read_csv(UCI_ADULT_URL, header=None, names=_adult_cols, skipinitialspace=True)
adult_df = adult_df.replace("?", pd.NA)
adult_df["income"] = adult_df["income"].str.strip()
save(adult_df, "adult_income.csv")

# ---------------------------------------------------------------------------
# REGRESSION
# ---------------------------------------------------------------------------

# California housing (downloaded by scikit-learn on first use, then cached).
housing = fetch_california_housing(as_frame=True)
housing_df = housing.frame.copy()
housing_df = housing_df.rename(columns={"MedHouseVal": "price"})
save(housing_df, "housing.csv")

diabetes = load_diabetes(as_frame=True)
diabetes_df = diabetes.frame.copy()
diabetes_df = diabetes_df.rename(columns={"target": "disease_progression"})
save(diabetes_df, "diabetes.csv")

wine_quality_df = pd.read_csv(UCI_WINE_QUALITY_RED_URL, sep=";")
save(wine_quality_df, "wine_quality_red.csv")

print("\nDone.")
