"""
generic_train.py — AutoML training on generic_prep.py's cleaned output.

Runs FLAML's AutoML over a short time budget to pick and tune a model for
ANY dataset, using generic_prep.py's cleaned CSV + task-type detection
instead of a hand-picked algorithm per dataset.

Usage:
    python generic_train.py <clean_csv_path> <target_col> [--task-type classification|regression]
                             [--time-budget 30] [--out-dir DIR] [--experiment NAME]

If --task-type is omitted, it is auto-detected from the target column using
the exact same detect_task_type() logic generic_prep.py used when it built
the cleaned CSV.

Outputs (into --out-dir, default: same folder as the input CSV):
    <stem>_model.pkl   the fitted FLAML AutoML object (call .predict(X))
    <stem>_meta.json   feature_cols, task_type, target_col, best_model, score

Also logs params/metrics/model to MLflow (sqlite:///mlflow.db, same as the
rest of this project).
"""

import argparse
import json
import pathlib
import pickle
import sys
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from flaml import AutoML
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from generic_prep import detect_task_type

ROOT = pathlib.Path(__file__).parent


def train(
    csv_path: str,
    target_col: str,
    task_type: str | None = None,
    time_budget: int = 30,
    out_dir: str | None = None,
    experiment: str = "generic-automl",
) -> dict:
    src = pathlib.Path(csv_path)
    df = pd.read_csv(src)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {list(df.columns)}")

    # --- task type: reuse generic_prep.py's detection unless overridden ---
    if task_type is None:
        task_type, reason = detect_task_type(df[target_col])
        print(f"Task type auto-detected: {task_type.upper()} ({reason})")
    else:
        if task_type not in ("classification", "regression"):
            raise ValueError("task_type must be 'classification' or 'regression'")
        print(f"Task type given explicitly: {task_type.upper()}")

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    stratify = y if task_type == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except ValueError:
        # Falls back when a class has too few members to stratify.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # --- FLAML AutoML: try several model families under a time budget -----
    metric = "accuracy" if task_type == "classification" else "r2"
    automl = AutoML()
    print(f"\nRunning FLAML AutoML (task={task_type}, metric={metric}, "
          f"time_budget={time_budget}s)...")
    t0 = time.time()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task=task_type,
        metric=metric,
        time_budget=time_budget,
        seed=42,
        mlflow_logging=False,  # we log to MLflow ourselves below
        verbose=0,
    )
    elapsed = time.time() - t0

    best_model_name = automl.best_estimator
    best_config = automl.best_config

    # --- Evaluate on held-out test set -------------------------------------
    y_pred = automl.predict(X_test)
    if task_type == "classification":
        score = float(accuracy_score(y_test, y_pred))
        metrics = {"accuracy": score}
    else:
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
        score = r2
        metrics = {"r2": r2, "rmse": rmse}

    # --- MLflow logging -----------------------------------------------------
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("time_budget", time_budget)
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        for k, v in best_config.items():
            mlflow.log_param(f"cfg_{k}", v)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        # cloudpickle instead of MLflow's default skops serialization: FLAML can
        # pick XGBoost/LightGBM estimators that skops refuses to trust by default.
        mlflow.sklearn.log_model(
            automl.model.estimator,
            "model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        run_id = run.info.run_id

    # --- Save model + feature list + task type ------------------------------
    out_dir_path = pathlib.Path(out_dir) if out_dir else src.parent
    out_dir_path.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    model_path = out_dir_path / f"{stem}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(automl, f)

    meta = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "task_type": task_type,
        "best_model": best_model_name,
        "best_config": best_config,
        "metrics": metrics,
        "mlflow_run_id": run_id,
    }
    meta_path = out_dir_path / f"{stem}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # --- Summary --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Task type:       {task_type}")
    print(f"Best model:      {best_model_name}")
    print(f"Best config:     {best_config}")
    print(f"AutoML runtime:  {elapsed:.1f}s")
    if task_type == "classification":
        print(f"Accuracy:        {metrics['accuracy']:.4f}")
    else:
        print(f"R^2:             {metrics['r2']:.4f}")
        print(f"RMSE:            {metrics['rmse']:.4f}")
    print(f"MLflow run:      {run_id}")
    print(f"Saved model:     {model_path}")
    print(f"Saved metadata:  {meta_path}")

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a FLAML AutoML model on generic_prep.py's cleaned CSV output."
    )
    parser.add_argument("csv_path", help="Path to the cleaned CSV file (generic_prep.py output)")
    parser.add_argument("target_col", help="Name of the target column")
    parser.add_argument(
        "--task-type",
        choices=["classification", "regression"],
        default=None,
        help="Override auto-detection (default: auto-detect via generic_prep.detect_task_type)",
    )
    parser.add_argument("--time-budget", type=int, default=30, help="FLAML time budget in seconds (default 30)")
    parser.add_argument("--out-dir", default=None, help="Where to save model/meta (default: alongside input CSV)")
    parser.add_argument("--experiment", default="generic-automl", help="MLflow experiment name")
    args = parser.parse_args()

    try:
        train(
            args.csv_path,
            args.target_col,
            task_type=args.task_type,
            time_budget=args.time_budget,
            out_dir=args.out_dir,
            experiment=args.experiment,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
