"""
Train the Titanic RandomForest locally with MLflow experiment tracking.

Default (no args) — identical to original behaviour, also updates
serving/model.pkl and serving/feature_cols.json used by the Docker image:
    python serving/train_model.py

Hyperparameter experiments (logs to MLflow, does NOT overwrite serving artefacts):
    python serving/train_model.py --n-estimators 200 --max-depth 8
    python serving/train_model.py --n-estimators 50 --max-depth 3 --min-samples-split 5

View runs:
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # from repo root
    open http://localhost:5000
"""

import argparse
import json
import pathlib
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train Titanic RF + log to MLflow.")
parser.add_argument("--n-estimators",      type=int,  default=100,
                    help="Number of trees (default 100)")
parser.add_argument("--max-depth",         type=int,  default=None,
                    help="Max tree depth; omit for unlimited (default)")
parser.add_argument("--min-samples-split", type=int,  default=2,
                    help="Min samples to split a node (default 2)")
parser.add_argument("--experiment",        default="titanic-hyperparameter-tuning",
                    help="MLflow experiment name")
args = parser.parse_args()

_is_default = (
    args.n_estimators == 100
    and args.max_depth is None
    and args.min_samples_split == 2
)

# ---------------------------------------------------------------------------
# 1. Load data  (local CSV first, URL fallback)
# ---------------------------------------------------------------------------

_local_csv = ROOT / "data" / "titanic.csv"
if _local_csv.exists():
    df = pd.read_csv(_local_csv)
    print(f"Loaded {len(df)} rows from {_local_csv.relative_to(ROOT)}")
else:
    _url = (
        "https://raw.githubusercontent.com/pandas-dev/pandas"
        "/main/doc/data/titanic.csv"
    )
    df = pd.read_csv(_url)
    print(f"Loaded {len(df)} rows from URL")

# Identical preprocessing to data_prep KFP component
cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
df = df[cols].copy()
df["Age"] = df["Age"].fillna(df["Age"].median())
df = df.dropna(subset=["Embarked"])
df["Sex"]      = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

FEATURE_COLS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------

X = df[FEATURE_COLS]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    min_samples_split=args.min_samples_split,
    random_state=42,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc  = float(accuracy_score(y_test, y_pred))
prec = float(precision_score(y_test, y_pred))
rec  = float(recall_score(y_test, y_pred))

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")

# ---------------------------------------------------------------------------
# 3. MLflow logging
# ---------------------------------------------------------------------------

# MLflow 3.x requires a database backend (file-store is deprecated).
# SQLite keeps everything in a single local file — no server needed for logging.
# Use sqlite:///path format; as_posix() gives forward-slash Windows paths
# that SQLAlchemy accepts (e.g. sqlite:///C:/Users/.../mlflow.db).
mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
mlflow.set_experiment(args.experiment)

with mlflow.start_run():
    # Params
    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         args.max_depth if args.max_depth is not None else "unlimited")
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("test_rows",         len(y_test))
    mlflow.log_param("train_rows",        len(X_train))

    # Metrics
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)

    # Model artifact
    mlflow.sklearn.log_model(clf, "random_forest")

    _run_id = mlflow.active_run().info.run_id
    print(f"MLflow run: {_run_id}")

# ---------------------------------------------------------------------------
# 4. Save serving artefacts (only for the default/production run)
# ---------------------------------------------------------------------------

if _is_default:
    _model_path    = HERE / "model.pkl"
    _features_path = HERE / "feature_cols.json"
    with open(_model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved {_model_path}")
    with open(_features_path, "w") as f:
        json.dump(FEATURE_COLS, f)
    print(f"Saved {_features_path}")
else:
    print("Non-default hyperparams — serving/model.pkl not overwritten.")
