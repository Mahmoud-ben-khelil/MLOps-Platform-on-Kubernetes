"""
KFP component: generate_reference_stats
  - Runs inside a pipeline pod as the first step.
  - Applies the SAME preprocessing as data_prep.
  - Saves per-column distribution statistics as a JSON artifact that
    drift_check will later compare new data against.

Also runnable as a local script to produce components/reference_stats.json:
    python components/generate_reference_stats.py
"""

import pathlib

from kfp import dsl
from kfp.dsl import Artifact, Output

HERE = pathlib.Path(__file__).parent

_URL = (
    "https://raw.githubusercontent.com/pandas-dev/pandas"
    "/main/doc/data/titanic.csv"
)
_FEATURE_COLS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


# ---------------------------------------------------------------------------
# KFP component  (body must be self-contained — no module-level imports)
# ---------------------------------------------------------------------------

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas==2.2.2", "numpy==1.26.4"],
)
def generate_reference_stats(reference_stats: Output[Artifact]) -> None:
    """Download Titanic CSV, preprocess identically to data_prep, save stats."""
    import json
    import pandas as pd

    url = (
        "https://raw.githubusercontent.com/pandas-dev/pandas"
        "/main/doc/data/titanic.csv"
    )
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    df = pd.read_csv(url)
    cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
    df = df[cols].copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df = df.dropna(subset=["Embarked"])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    stats: dict = {"n_rows": len(df), "columns": {}}
    for col in feature_cols:
        series = df[col].dropna()
        stats["columns"][col] = {
            "dtype": str(df[col].dtype),
            "missing_rate": float(df[col].isna().mean()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            # Full sample stored so drift_check can run a proper KS test.
            "values": [float(v) for v in series.tolist()],
        }

    with open(reference_stats.path, "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"Reference stats written ({len(df)} rows, {len(feature_cols)} cols)")
    for col in feature_cols:
        s = stats["columns"][col]
        print(f"  {col:10s}: mean={s['mean']:.4f}  std={s['std']:.4f}  "
              f"missing={s['missing_rate']:.1%}")


# ---------------------------------------------------------------------------
# Local script  (uses module-level constants — runs outside a pod)
# ---------------------------------------------------------------------------

def _build_stats_locally() -> dict:
    import json
    import pandas as pd

    df = pd.read_csv(_URL)
    cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
    df = df[cols].copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df = df.dropna(subset=["Embarked"])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    stats: dict = {"n_rows": len(df), "columns": {}}
    for col in _FEATURE_COLS:
        series = df[col].dropna()
        stats["columns"][col] = {
            "dtype": str(df[col].dtype),
            "missing_rate": float(df[col].isna().mean()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "values": [float(v) for v in series.tolist()],
        }
    return stats


if __name__ == "__main__":
    import json

    stats = _build_stats_locally()
    out = HERE / "reference_stats.json"
    with open(out, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Written: {out}  ({stats['n_rows']} rows)")
    for col, s in stats["columns"].items():
        print(f"  {col:10s}: mean={s['mean']:.4f}  std={s['std']:.4f}  "
              f"missing={s['missing_rate']:.1%}")
