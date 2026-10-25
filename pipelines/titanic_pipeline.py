"""
Titanic pipeline: generate_reference_stats + data_prep -> drift_check -> (train)

        generate_reference_stats --+
                                   +--> drift_check --> [dsl.If passed] --> train
        data_prep ------------------+

The drift_check gate sets validation_passed=True/False.
If validation fails, the train step is SKIPPED and the pipeline shows the
drift report in the KFP UI as a Markdown artifact.

Run this file to compile pipeline.yaml:
    python pipelines/titanic_pipeline.py

Submit a clean run (validation passes, train runs):
    python pipelines/titanic_pipeline.py --submit

Submit a drifted run (validation fails, train is SKIPPED):
    python pipelines/titanic_pipeline.py --drift
"""

import argparse
import pathlib
import sys

# Make `components/` importable when this script is run from anywhere.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from components.drift_check import drift_check                            # noqa: E402
from components.generate_reference_stats import generate_reference_stats  # noqa: E402
from kfp import dsl, compiler
from kfp.dsl import Dataset, Input, Metrics, Model, Output


# ---------------------------------------------------------------------------
# Step 2 — data prep
# ---------------------------------------------------------------------------

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas==2.2.2", "numpy==1.26.4"],
)
def data_prep(processed: Output[Dataset], drift_mode: bool = False) -> None:
    """
    Download Titanic CSV, clean and encode.

    drift_mode=False (default): standard preprocessing, used for clean runs.
    drift_mode=True:  applies synthetic drift before saving:
      - Age  +30  (mean: ~29 -> ~59, shift ~2.3 std devs)
      - Fare *8 +100  (mean: ~32 -> ~356, shift >>2 std devs)
    The reference stats always come from the clean distribution, so both
    perturbations are well outside the KS and mean-shift thresholds.
    """
    import pandas as pd

    url = (
        "https://raw.githubusercontent.com/pandas-dev/pandas"
        "/main/doc/data/titanic.csv"
    )
    df = pd.read_csv(url)

    cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
    df = df[cols].copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df = df.dropna(subset=["Embarked"])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    if drift_mode:
        # Age: shift every passenger 30 years older.
        # Reference mean ~29, std ~13 -> new mean ~59 -> shift ~2.3 std devs.
        df["Age"] = df["Age"] + 30

        # Fare: scale by 8x and add 100 flat.
        # Reference mean ~32 -> new mean ~356 -> shift well above 2 std devs.
        df["Fare"] = df["Fare"] * 8 + 100

        print("drift_mode=True: Age +30 and Fare *8+100 applied to all rows")

    df.to_csv(processed.path, index=False)
    print(f"Saved {len(df)} rows  drift_mode={drift_mode}")


# ---------------------------------------------------------------------------
# Step 4 — train  (only runs when validation gate passes)
# ---------------------------------------------------------------------------

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas==2.2.2", "scikit-learn==1.5.0"],
)
def train(
    processed: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 100,
    max_depth: int = 0,
    min_samples_split: int = 2,
) -> None:
    """Train RandomForestClassifier and log accuracy + hyperparameters."""
    import pickle
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(processed.path)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # max_depth=0 is the sentinel for "no limit" (None), since KFP params
    # cannot be Optional[int]. Convert here so RF receives the correct type.
    _max_depth = None if max_depth == 0 else max_depth

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=_max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    acc = float(accuracy_score(y_test, clf.predict(X_test)))
    print(f"Accuracy: {acc:.4f}")
    metrics.log_metric("accuracy", round(acc, 4))
    metrics.log_metric("n_estimators", n_estimators)
    metrics.log_metric("max_depth", max_depth)
    metrics.log_metric("min_samples_split", min_samples_split)
    metrics.log_metric("test_size", len(y_test))

    with open(model.path, "wb") as f:
        pickle.dump(clf, f)
    print(
        f"Model saved  n_estimators={n_estimators} "
        f"max_depth={_max_depth} min_samples_split={min_samples_split}"
    )


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

@dsl.pipeline(
    name="titanic-pipeline",
    description=(
        "Titanic: data prep -> KS drift check -> conditional train. "
        "Set drift_mode=True to inject Age/Fare drift and watch the gate close."
    ),
)
def titanic_pipeline(
    drift_mode: bool = False,
    missing_threshold: float = 0.20,
    ks_pvalue_threshold: float = 0.05,
    mean_shift_threshold: float = 2.0,
    n_estimators: int = 100,
    max_depth: int = 0,
    min_samples_split: int = 2,
) -> None:
    # generate_reference_stats and data_prep run in parallel (no dependency).
    # generate_reference_stats always uses clean data as the baseline.
    ref_task = generate_reference_stats()
    prep_task = data_prep(drift_mode=drift_mode)

    # drift_check compares prep output against the clean reference.
    validate_task = drift_check(
        processed=prep_task.outputs["processed"],
        reference_stats=ref_task.outputs["reference_stats"],
        missing_threshold=missing_threshold,
        ks_pvalue_threshold=ks_pvalue_threshold,
        mean_shift_threshold=mean_shift_threshold,
    )

    # drift_check has two outputs (report artifact + bool scalar).
    # KFP names the bool scalar "Output" when mixed with artifact outputs.
    # When drift_mode=True the bool is False -> condition not met -> train skipped.
    with dsl.If(validate_task.outputs["Output"] == True, name="validation-gate"):  # noqa: E712
        train(
            processed=prep_task.outputs["processed"],
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )


# ---------------------------------------------------------------------------
# Compile / submit
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile and optionally submit the Titanic KFP pipeline."
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit a clean run (drift_mode=False) to KFP at http://localhost:8082",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help=(
            "Submit a DRIFTED run (drift_mode=True) — Age+30, Fare*8+100. "
            "Drift-check fires, validation gate closes, train is SKIPPED."
        ),
    )
    args = parser.parse_args()

    output_file = "pipelines/titanic_pipeline.yaml"
    compiler.Compiler().compile(titanic_pipeline, output_file)
    print(f"Compiled -> {output_file}")

    if args.submit or args.drift:
        import kfp

        client = kfp.Client(host="http://localhost:8082")

        if args.drift:
            run_name = "titanic-DRIFTED-gate-demo"
            run_args = {
                "drift_mode": True,
                "ks_pvalue_threshold": 0.05,
                "mean_shift_threshold": 2.0,
            }
            print("Submitting DRIFTED run — expect Age + Fare to fail the KS test.")
        else:
            run_name = "titanic-clean-run"
            run_args = {"drift_mode": False}
            print("Submitting clean run — expect all columns to pass.")

        run = client.create_run_from_pipeline_func(
            titanic_pipeline,
            arguments=run_args,
            experiment_name="titanic",
            run_name=run_name,
        )
        print(f"Submitted run : {run.run_id}")
        print(f"View at       : http://localhost:8082/#/runs/details/{run.run_id}")
