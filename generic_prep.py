"""
generic_prep.py — dataset-agnostic preprocessing.

Takes any CSV + a target column name and produces a cleaned, fully-numeric
feature table plus the target column, using generic rules instead of
Titanic-specific column names:

  1. Load CSV, list all columns.
  2. Auto-detect each feature as numeric vs categorical.
  3. Clean: median-fill numeric, mode-fill categorical; drop useless columns
     (all-unique ID-like columns, or columns with >50% missing).
  4. Encode categoricals to integers; drop high-cardinality categoricals
     instead of exploding them into hundreds of one-hot columns.
  5. Auto-detect classification vs regression from the target column.
  6. Print a summary and write the cleaned table to CSV.

Usage:
    python generic_prep.py <csv_path> <target_column> [--out OUT_PATH]
"""

import argparse
import pathlib
import sys

import pandas as pd

HIGH_CARDINALITY_COUNT = 50   # above this many unique values, a categorical
HIGH_CARDINALITY_RATIO = 0.5  # column is treated as free-text/ID-like, not
                               # a genuine category, and is dropped.
MISSING_THRESHOLD = 0.5       # drop columns missing more than this fraction.
CLASSIFICATION_MAX_DISTINCT = 20     # numeric targets with few distinct
CLASSIFICATION_MAX_DISTINCT_RATIO = 0.05  # values (absolute or relative to
                                           # row count) are treated as class
                                           # labels rather than a continuous
                                           # quantity to regress on.


def detect_column_type(series: pd.Series) -> str:
    """Return 'numeric' or 'categorical' for a single feature column."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def detect_task_type(target: pd.Series) -> tuple[str, str]:
    """Return (task_type, reason) for the target column."""
    n = len(target)
    n_distinct = target.nunique(dropna=True)

    if not pd.api.types.is_numeric_dtype(target):
        return "classification", (
            f"target dtype is non-numeric ({target.dtype}) -> treated as class labels"
        )

    if n_distinct <= CLASSIFICATION_MAX_DISTINCT or (n_distinct / n) <= CLASSIFICATION_MAX_DISTINCT_RATIO:
        return "classification", (
            f"target is numeric but only has {n_distinct} distinct values "
            f"out of {n} rows -> treated as class labels, not a continuous quantity"
        )

    return "regression", (
        f"target is numeric with {n_distinct} distinct values out of {n} rows "
        f"-> treated as a continuous quantity"
    )


def prep(csv_path: str, target_col: str, out_path: str | None = None) -> pd.DataFrame:
    src = pathlib.Path(csv_path)
    df = pd.read_csv(src)

    print(f"Loaded {src.name}: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}\n")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {list(df.columns)}")

    n_rows = len(df)
    target = df[target_col]
    feature_df = df.drop(columns=[target_col])

    dropped = {}      # column -> reason
    kept_numeric = []
    kept_categorical = []

    # --- Step 2+3: detect type, decide keep/drop -----------------------
    for col in feature_df.columns:
        series = feature_df[col]
        missing_ratio = series.isna().mean()

        if missing_ratio > MISSING_THRESHOLD:
            dropped[col] = f">{MISSING_THRESHOLD:.0%} missing ({missing_ratio:.0%})"
            continue

        n_unique = series.nunique(dropna=True)
        is_all_unique = n_unique >= n_rows - 1 and n_rows > 1  # ID-like column

        col_type = detect_column_type(series)

        if col_type == "categorical":
            if is_all_unique:
                dropped[col] = f"all-unique ({n_unique}/{n_rows}) -> ID/name-like column"
                continue
            if n_unique > HIGH_CARDINALITY_COUNT and (n_unique / n_rows) > HIGH_CARDINALITY_RATIO:
                dropped[col] = (
                    f"high-cardinality categorical ({n_unique} unique values) "
                    "-> would explode into too many encoded columns"
                )
                continue
            kept_categorical.append(col)
        else:
            if is_all_unique:
                dropped[col] = f"all-unique ({n_unique}/{n_rows}) -> ID-like numeric column"
                continue
            kept_numeric.append(col)

    # --- Step 3: fill missing values ------------------------------------
    clean = feature_df[kept_numeric + kept_categorical].copy()

    for col in kept_numeric:
        median = clean[col].median()
        n_missing = clean[col].isna().sum()
        if n_missing:
            clean[col] = clean[col].fillna(median)
            print(f"Filled {n_missing} missing values in numeric '{col}' with median={median}")

    for col in kept_categorical:
        mode = clean[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "missing"
        n_missing = clean[col].isna().sum()
        if n_missing:
            clean[col] = clean[col].fillna(fill_value)
            print(f"Filled {n_missing} missing values in categorical '{col}' with mode='{fill_value}'")

    # --- Step 4: encode categoricals to integers ------------------------
    encodings = {}
    for col in kept_categorical:
        categories = sorted(clean[col].astype(str).unique())
        mapping = {cat: i for i, cat in enumerate(categories)}
        clean[col] = clean[col].astype(str).map(mapping)
        encodings[col] = mapping

    # --- Step 5: task type from target -----------------------------------
    task_type, reason = detect_task_type(target)

    # Encode a classification target if it's non-numeric text labels.
    target_clean = target.copy()
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(target_clean):
        classes = sorted(target_clean.dropna().unique())
        target_mapping = {c: i for i, c in enumerate(classes)}
        target_clean = target_clean.map(target_mapping)
        print(f"\nEncoded target classes: {target_mapping}")

    result = clean.copy()
    result[target_col] = target_clean.values

    # --- Step 6: summary ---------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Columns kept ({len(kept_numeric) + len(kept_categorical)}):")
    for col in kept_numeric:
        print(f"  - {col:<20} numeric")
    for col in kept_categorical:
        print(f"  - {col:<20} categorical -> encoded ({len(encodings[col])} classes)")

    print(f"\nColumns dropped ({len(dropped)}):")
    if dropped:
        for col, reason_dropped in dropped.items():
            print(f"  - {col:<20} {reason_dropped}")
    else:
        print("  (none)")

    print(f"\nTask type: {task_type.upper()}")
    print(f"Reason: {reason}")
    print(f"\nFinal shape: {result.shape[0]} rows x {result.shape[1]} columns "
          f"({result.shape[1] - 1} features + target)")

    out = pathlib.Path(out_path) if out_path else src.with_name(src.stem + "_clean.csv")
    result.to_csv(out, index=False)
    print(f"\nSaved cleaned data -> {out}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic dataset preprocessing for any CSV.")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument("target_col", help="Name of the target column")
    parser.add_argument("--out", default=None, help="Output path for cleaned CSV (default: <input>_clean.csv)")
    args = parser.parse_args()

    try:
        prep(args.csv_path, args.target_col, args.out)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
