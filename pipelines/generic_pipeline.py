"""
Generic pipeline: dataset-agnostic sibling of titanic_pipeline.py.

    dsl.importer(csv_url) --+--> generic_reference_stats --+
                             |                              +--> generic_drift_check --> [dsl.If passed] --> generic_train
                             +--> generic_data_prep --------+

Every step is driven by two pipeline parameters instead of Titanic-specific
column names: `csv_url` (where the dataset lives) and `target_column` (what
to predict). generic_reference_stats and generic_data_prep both clean the
SAME imported CSV artifact — the reference is always derived from whatever
dataset was actually passed in, never a hardcoded baseline. drift_mode=True
perturbs every numeric feature column generically (no column names baked in)
so the KS gate can be demoed on any dataset, mirroring Titanic's Age+30/Fare*8+100.

Run this file to compile pipelines/generic_pipeline.yaml:
    $env:PYTHONUTF8="1"; python pipelines/generic_pipeline.py

Submit a clean run for Iris (validation passes, train runs):
    $env:PYTHONUTF8="1"; python pipelines/generic_pipeline.py --submit `
        --csv-url "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv" `
        --target-column species

Submit a drifted run (validation fails, train is SKIPPED):
    $env:PYTHONUTF8="1"; python pipelines/generic_pipeline.py --drift `
        --csv-url "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv" `
        --target-column species
"""

import argparse

from kfp import dsl, compiler
from kfp.dsl import Artifact, Dataset, Input, Markdown, Metrics, Model, Output


# ---------------------------------------------------------------------------
# Step 0 — fetch the CSV into a Dataset artifact.
#
# dsl.importer() is KFP's built-in way to bring an external URI in as an
# artifact without routing the bytes through pipeline parameters, but its
# importer image doesn't reliably support plain https:// URIs on every KFP
# install (this one included -- confirmed by a live test run where the
# importer step failed immediately). A regular component doing the download
# itself is the proven-working path: titanic_pipeline's data_prep step
# already does exactly this (pd.read_csv(url) from inside a pod) on this
# same cluster. Same artifact-passing contract downstream either way --
# only the ingestion mechanism changes.
# ---------------------------------------------------------------------------

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas==2.2.2"],
)
def generic_fetch_csv(csv_url: str, csv: Output[Dataset]) -> None:
    import pandas as pd

    df = pd.read_csv(csv_url)
    df.to_csv(csv.path, index=False)
    print(f"Fetched {csv_url} -> {len(df)} rows, {len(df.columns)} columns")


# ---------------------------------------------------------------------------
# Step 1a — reference stats (derived from the same CSV artifact, never a
# separate hardcoded baseline)
# ---------------------------------------------------------------------------

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas==2.2.2", "numpy==1.26.4"],
)
def generic_reference_stats(
    csv: Input[Dataset],
    target_column: str,
    reference_stats: Output[Artifact],
) -> None:
    """Clean the imported CSV the same way generic_data_prep will, and record
    per-feature mean/std/sample so generic_drift_check has a baseline to
    compare future runs against."""
    import json
    import pandas as pd

    df = pd.read_csv(csv.path)
    n_rows = len(df)
    feature_cols_in = [c for c in df.columns if c != target_column]

    kept_numeric, kept_categorical = [], []
    for col in feature_cols_in:
        s = df[col]
        if float(s.isna().mean()) > 0.5:
            continue
        n_unique = s.nunique(dropna=True)
        is_id_like = n_unique >= n_rows - 1 and n_rows > 1
        if pd.api.types.is_numeric_dtype(s):
            if is_id_like:
                continue
            kept_numeric.append(col)
        else:
            if is_id_like or (n_unique > 50 and (n_unique / n_rows) > 0.5):
                continue
            kept_categorical.append(col)

    clean = pd.DataFrame(index=df.index)
    for col in kept_numeric:
        clean[col] = df[col].fillna(df[col].median())
    for col in kept_categorical:
        mode = df[col].mode(dropna=True)
        fill = mode.iloc[0] if not mode.empty else "missing"
        vals = df[col].fillna(fill).astype(str)
        mapping = {c: i for i, c in enumerate(sorted(vals.unique()))}
        clean[col] = vals.map(mapping)

    stats = {"n_rows": len(clean), "columns": {}}
    for col in clean.columns:
        series = clean[col].dropna()
        stats["columns"][col] = {
            "mean": float(series.mean()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "values": [float(v) for v in series.tolist()],
        }

    with open(reference_stats.path, "w") as fh:
        json.dump(stats, fh)

    print(f"Reference stats written ({len(clean)} rows, {len(clean.columns)} feature cols)")
    for col, s in stats["columns"].items():
        print(f"  {col:20s} mean={s['mean']:.4f}  std={s['std']:.4f}")


# ---------------------------------------------------------------------------
# Step 1b — data prep (generic_prep.py logic, self-contained for the pod)
# ---------------------------------------------------------------------------

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas==2.2.2", "numpy==1.26.4"],
)
def generic_data_prep(
    csv: Input[Dataset],
    target_column: str,
    processed: Output[Dataset],
    meta: Output[Artifact],
    drift_mode: bool = False,
) -> None:
    """Clean + encode any CSV: median/mode fill, drop ID-like and
    high-cardinality columns, encode categoricals, detect task type."""
    import json
    import pandas as pd

    df = pd.read_csv(csv.path)
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not in columns {list(df.columns)}")

    n_rows = len(df)
    feature_cols_in = [c for c in df.columns if c != target_column]

    kept_numeric, kept_categorical, dropped = [], [], {}
    for col in feature_cols_in:
        s = df[col]
        missing_ratio = float(s.isna().mean())
        if missing_ratio > 0.5:
            dropped[col] = f">50% missing ({missing_ratio:.0%})"
            continue

        n_unique = s.nunique(dropna=True)
        is_id_like = n_unique >= n_rows - 1 and n_rows > 1

        if pd.api.types.is_numeric_dtype(s):
            if is_id_like:
                dropped[col] = f"all-unique ({n_unique}/{n_rows}) -> ID-like numeric column"
                continue
            kept_numeric.append(col)
        else:
            if is_id_like:
                dropped[col] = f"all-unique ({n_unique}/{n_rows}) -> ID/name-like column"
                continue
            if n_unique > 50 and (n_unique / n_rows) > 0.5:
                dropped[col] = f"high-cardinality categorical ({n_unique} unique values)"
                continue
            kept_categorical.append(col)

    clean = df[kept_numeric + kept_categorical].copy()
    for col in kept_numeric:
        clean[col] = clean[col].fillna(clean[col].median())
    for col in kept_categorical:
        mode = clean[col].mode(dropna=True)
        fill = mode.iloc[0] if not mode.empty else "missing"
        clean[col] = clean[col].fillna(fill).astype(str)
        mapping = {c: i for i, c in enumerate(sorted(clean[col].unique()))}
        clean[col] = clean[col].map(mapping)

    # --- task type detection (mirrors generic_prep.detect_task_type) -------
    target = df[target_column]
    n_distinct = target.nunique(dropna=True)
    if not pd.api.types.is_numeric_dtype(target):
        task_type = "classification"
        classes = sorted(target.dropna().unique())
        mapping = {c: i for i, c in enumerate(classes)}
        target_clean = target.map(mapping)
    elif n_distinct <= 20 or (n_distinct / n_rows) <= 0.05:
        task_type = "classification"
        target_clean = target
    else:
        task_type = "regression"
        target_clean = target

    if drift_mode:
        # Generic perturbation: scale + shift every numeric feature column so
        # the KS gate fires regardless of which dataset/columns are in play,
        # mirroring Titanic's Age+30 / Fare*8+100 without hardcoding names.
        for col in kept_numeric:
            std = clean[col].std()
            clean[col] = clean[col] * 3 + (std if std > 0 else 1.0) * 5
        print(f"drift_mode=True: perturbed numeric columns {kept_numeric} (x3, +5 std)")

    result = clean.copy()
    result[target_column] = target_clean.values
    result.to_csv(processed.path, index=False)

    meta_dict = {
        "feature_cols": kept_numeric + kept_categorical,
        "target_col": target_column,
        "task_type": task_type,
        "dropped_cols": dropped,
    }
    with open(meta.path, "w") as fh:
        json.dump(meta_dict, fh)

    print(f"Saved {len(result)} rows, {len(kept_numeric + kept_categorical)} features, "
          f"task={task_type}, drift_mode={drift_mode}")
    if dropped:
        print("Dropped columns:")
        for c, r in dropped.items():
            print(f"  {c}: {r}")


# ---------------------------------------------------------------------------
# Step 2 — drift check (same KS + mean-shift logic as drift_check.py, but the
# expected schema is read from reference_stats instead of hardcoded)
# ---------------------------------------------------------------------------

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas==2.2.2", "numpy==1.26.4", "scipy==1.13.0"],
)
def generic_drift_check(
    processed: Input[Dataset],
    reference_stats: Input[Artifact],
    report: Output[Markdown],
    missing_threshold: float = 0.20,
    ks_pvalue_threshold: float = 0.05,
    mean_shift_threshold: float = 2.0,
) -> bool:
    """Validate new data against reference statistics computed from the same
    dataset. Returns True if all checks pass (safe to train)."""
    import json
    import pandas as pd
    from scipy import stats as scipy_stats

    df = pd.read_csv(processed.path)
    with open(reference_stats.path) as fh:
        ref = json.load(fh)

    feature_cols = list(ref["columns"].keys())
    issues: list = []

    # 1. SCHEMA CHECK -- dynamic: every reference column must be present
    schema_rows = []
    for col in feature_cols:
        ok = col in df.columns
        if not ok:
            issues.append(f"SCHEMA: column '{col}' is missing")
        schema_rows.append(f"| {col} | {'[OK]' if ok else '[MISSING]'} |")

    # 2. DATA QUALITY CHECK
    quality_rows = []
    for col in df.columns:
        rate = float(df[col].isna().mean())
        ok = rate <= missing_threshold
        if not ok:
            issues.append(f"QUALITY: '{col}' has {rate:.1%} missing (threshold {missing_threshold:.0%})")
        quality_rows.append(f"| {col} | {rate:.1%} | {'[OK]' if ok else '[EXCEEDS THRESHOLD]'} |")

    # 3. DRIFT DETECTION -- KS test + mean-shift on every reference column
    col_results: dict = {}
    drift_rows = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        ref_col = ref["columns"][col]
        ref_values = ref_col["values"]
        new_values = df[col].dropna().tolist()
        if len(ref_values) < 2 or len(new_values) < 2:
            continue

        ks_stat_raw, p_value_raw = scipy_stats.ks_2samp(ref_values, new_values)
        ks_stat, p_value = float(ks_stat_raw), float(p_value_raw)

        ref_mean, ref_std = float(ref_col["mean"]), float(ref_col["std"])
        new_mean = float(df[col].mean())
        shift_stds = abs(new_mean - ref_mean) / ref_std if ref_std > 0 else 0.0

        ks_drift = bool(p_value < ks_pvalue_threshold)
        mean_drift = bool(shift_stds > mean_shift_threshold)
        drifted = ks_drift or mean_drift

        if ks_drift:
            issues.append(f"DRIFT: '{col}' KS p={p_value:.4f} < {ks_pvalue_threshold} (stat={ks_stat:.4f})")
        if mean_drift:
            issues.append(f"DRIFT: '{col}' mean shifted {shift_stds:.2f} std devs "
                           f"(ref={ref_mean:.4f} -> new={new_mean:.4f})")

        col_results[col] = {
            "ks_statistic": round(ks_stat, 4), "ks_pvalue": round(p_value, 4),
            "ref_mean": round(ref_mean, 4), "new_mean": round(new_mean, 4),
            "mean_shift_stds": round(shift_stds, 4), "drifted": drifted,
        }
        status = "[DRIFT]" if drifted else "[OK]"
        drift_rows.append(f"| {col} | {ks_stat:.4f} | {p_value:.4f} | "
                           f"{ref_mean:.4f} | {new_mean:.4f} | {shift_stds:.2f} std | {status} |")

    validation_passed = len(issues) == 0
    status_badge = "PASSED" if validation_passed else f"FAILED ({len(issues)} issues)"

    md_lines = [
        "# Data Validation Report", "",
        f"**Status**: {status_badge}  ",
        f"**Rows**: {len(df)}  |  **Feature columns checked**: {len(feature_cols)}  ",
        f"**Reference rows**: {ref.get('n_rows', '?')}", "",
    ]
    if issues:
        md_lines += ["## Issues Found", ""] + [f"- {i}" for i in issues] + [""]
    md_lines += [
        "## 1 - Schema Check", "", "| Column | Status |", "|--------|--------|",
        *schema_rows, "",
        "## 2 - Data Quality (missing values)", "",
        f"Threshold: **{missing_threshold:.0%}**", "",
        "| Column | Missing Rate | Status |", "|--------|-------------|--------|",
        *quality_rows, "",
        "## 3 - Drift Detection", "",
        f"KS p-value threshold: **{ks_pvalue_threshold}**  |  "
        f"Mean-shift threshold: **{mean_shift_threshold} std devs**", "",
        "| Column | KS stat | p-value | Ref mean | New mean | Shift | Status |",
        "|--------|---------|---------|----------|----------|-------|--------|",
        *drift_rows, "",
        "## Full JSON", "", "```json",
        json.dumps({"validation_passed": validation_passed, "issues": issues,
                     "column_drift": col_results}, indent=2),
        "```",
    ]
    with open(report.path, "w") as fh:
        fh.write("\n".join(md_lines))

    print("=" * 60)
    print(f"VALIDATION: {status_badge}")
    for issue in issues:
        print(f"  [x] {issue}")
    print("=" * 60)

    return validation_passed


# ---------------------------------------------------------------------------
# Step 3 — train (FLAML AutoML, auto-detects task type from meta artifact)
# ---------------------------------------------------------------------------

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas==2.2.2", "numpy==1.26.4", "scikit-learn==1.5.0", "flaml[automl]==2.3.5"],
)
def generic_train(
    processed: Input[Dataset],
    meta: Input[Artifact],
    model: Output[Model],
    metrics: Output[Metrics],
    time_budget: int = 30,
) -> None:
    """Run FLAML AutoML on the cleaned data. Task type + feature list come
    from generic_data_prep's meta artifact -- no dataset-specific code here."""
    import json
    import pickle
    from flaml import AutoML
    import pandas as pd
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(processed.path)
    with open(meta.path) as fh:
        meta_dict = json.load(fh)

    target_col = meta_dict["target_col"]
    task_type = meta_dict["task_type"]
    feature_cols = meta_dict["feature_cols"]

    X = df[feature_cols]
    y = df[target_col]

    stratify = y if task_type == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    flaml_metric = "accuracy" if task_type == "classification" else "r2"
    automl = AutoML()
    automl.fit(
        X_train=X_train, y_train=y_train,
        task=task_type, metric=flaml_metric,
        time_budget=time_budget, seed=42,
        mlflow_logging=False, verbose=0,
    )

    y_pred = automl.predict(X_test)
    if task_type == "classification":
        acc = float(accuracy_score(y_test, y_pred))
        metrics.log_metric("accuracy", round(acc, 4))
        print(f"Accuracy: {acc:.4f}")
    else:
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
        metrics.log_metric("r2", round(r2, 4))
        metrics.log_metric("rmse", round(rmse, 4))
        print(f"R2: {r2:.4f}  RMSE: {rmse:.4f}")

    metrics.log_metric("time_budget", time_budget)
    metrics.log_metric("train_rows", len(X_train))
    metrics.log_metric("test_rows", len(X_test))

    model.metadata["best_model"] = str(automl.best_estimator)
    model.metadata["task_type"] = task_type
    model.metadata["target_col"] = target_col
    model.metadata["feature_cols"] = feature_cols

    with open(model.path, "wb") as f:
        pickle.dump(automl, f)

    print(f"Task type: {task_type}")
    print(f"Best model (FLAML): {automl.best_estimator}")


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

@dsl.pipeline(
    name="generic-automl-pipeline",
    description=(
        "Any CSV -> generic cleaning -> KS drift gate -> FLAML AutoML train. "
        "Dataset-agnostic sibling of titanic_pipeline. Set drift_mode=True to "
        "perturb every numeric feature and watch the gate close."
    ),
)
def generic_pipeline(
    csv_url: str,
    target_column: str,
    drift_mode: bool = False,
    missing_threshold: float = 0.20,
    ks_pvalue_threshold: float = 0.05,
    mean_shift_threshold: float = 2.0,
    time_budget: int = 30,
) -> None:
    # Fetch the CSV once into a Dataset artifact; the bytes never pass through
    # pipeline parameters (which have tight size limits) -- each downstream
    # pod consumes the artifact directly.
    fetch_task = generic_fetch_csv(csv_url=csv_url)

    # generic_reference_stats and generic_data_prep both clean the SAME
    # fetched artifact and run in parallel (no dependency between them).
    ref_task = generic_reference_stats(
        csv=fetch_task.outputs["csv"],
        target_column=target_column,
    )
    prep_task = generic_data_prep(
        csv=fetch_task.outputs["csv"],
        target_column=target_column,
        drift_mode=drift_mode,
    )

    validate_task = generic_drift_check(
        processed=prep_task.outputs["processed"],
        reference_stats=ref_task.outputs["reference_stats"],
        missing_threshold=missing_threshold,
        ks_pvalue_threshold=ks_pvalue_threshold,
        mean_shift_threshold=mean_shift_threshold,
    )

    # drift_check has two outputs (report artifact + bool scalar); KFP names
    # the bool scalar "Output" when mixed with artifact outputs.
    with dsl.If(validate_task.outputs["Output"] == True, name="validation-gate"):  # noqa: E712
        generic_train(
            processed=prep_task.outputs["processed"],
            meta=prep_task.outputs["meta"],
            time_budget=time_budget,
        )


# ---------------------------------------------------------------------------
# Compile / submit
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile and optionally submit the generic AutoML KFP pipeline."
    )
    parser.add_argument("--submit", action="store_true", help="Submit a clean run to KFP at http://localhost:8082")
    parser.add_argument("--drift", action="store_true", help="Submit a DRIFTED run -- validation gate blocks training")
    parser.add_argument("--csv-url", default=None, help="URL the cluster can fetch the CSV from")
    parser.add_argument("--target-column", default=None, help="Name of the target column")
    parser.add_argument("--time-budget", type=int, default=30, help="FLAML time budget in seconds (default 30)")
    args = parser.parse_args()

    output_file = "pipelines/generic_pipeline.yaml"
    compiler.Compiler().compile(generic_pipeline, output_file)
    print(f"Compiled -> {output_file}")

    if args.submit or args.drift:
        if not args.csv_url or not args.target_column:
            raise SystemExit("--submit/--drift require --csv-url and --target-column")

        import kfp

        client = kfp.Client(host="http://localhost:8082")

        run_args = {
            "csv_url": args.csv_url,
            "target_column": args.target_column,
            "drift_mode": bool(args.drift),
            "time_budget": args.time_budget,
        }
        run_name = "generic-DRIFTED-gate-demo" if args.drift else "generic-clean-run"
        print(f"Submitting {'DRIFTED' if args.drift else 'clean'} run for target={args.target_column!r}...")

        run = client.create_run_from_pipeline_func(
            generic_pipeline,
            arguments=run_args,
            experiment_name="generic-automl",
            run_name=run_name,
        )
        print(f"Submitted run : {run.run_id}")
        print(f"View at       : http://localhost:8082/#/runs/details/{run.run_id}")
