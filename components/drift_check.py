"""
KFP component: drift_check

Runs BETWEEN data_prep and train. Takes the preprocessed Dataset artifact
and a reference statistics JSON artifact, then performs three checks:

  1. SCHEMA CHECK     - all 7 expected columns present, correct dtypes
  2. DATA QUALITY     - no column exceeds missing_threshold (default 20 %)
  3. DRIFT DETECTION  - per column:
       a. Kolmogorov-Smirnov test against reference distribution.
          p < ks_pvalue_threshold (default 0.05) -> drift flag
       b. Mean-shift check: |mean_new - mean_ref| > mean_shift_threshold x std_ref
          (default 2.0 std devs) -> drift flag

Why KS test?
  The KS statistic is the maximum absolute gap between the empirical CDFs of
  the two samples. The p-value is the probability of seeing that gap purely by
  chance if both samples come from the same distribution. Low p-value means
  the distributions are unlikely to be the same -> drift.
  Advantages over a simple mean comparison:
    - Non-parametric: no normality assumption.
    - Catches shifts in mean, variance, AND distribution shape.
    - Works on small samples (Titanic ~800 rows).

Output: a Markdown artifact rendered in the KFP UI + a boolean return value
used by the pipeline's dsl.If gate to decide whether to run training.
"""

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Markdown, Output


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scipy==1.13.0",
    ],
)
def drift_check(
    processed: Input[Dataset],
    reference_stats: Input[Artifact],
    report: Output[Markdown],
    missing_threshold: float = 0.20,
    ks_pvalue_threshold: float = 0.05,
    mean_shift_threshold: float = 2.0,
) -> bool:
    """
    Validate new data against reference statistics.

    Returns True if all checks pass (safe to train), False otherwise.
    The Markdown artifact is always written regardless of outcome.
    """
    import json
    import pandas as pd
    from scipy import stats as scipy_stats

    EXPECTED_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    EXPECTED_DTYPES = {
        "Pclass": "int64",
        "Sex": "int64",
        "Age": "float64",
        "SibSp": "int64",
        "Parch": "int64",
        "Fare": "float64",
        "Embarked": "int64",
    }

    # -----------------------------------------------------------------------
    # Load inputs
    # -----------------------------------------------------------------------
    df = pd.read_csv(processed.path)
    with open(reference_stats.path) as fh:
        ref = json.load(fh)

    issues: list = []
    col_results: dict = {}

    # -----------------------------------------------------------------------
    # 1. SCHEMA CHECK
    # -----------------------------------------------------------------------
    schema_rows: list = []
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            issues.append(f"SCHEMA: column '{col}' is missing")
            schema_rows.append(f"| {col} | [MISSING] | - |")
        else:
            actual_dtype = str(df[col].dtype)
            expected_dtype = EXPECTED_DTYPES[col]
            ok = actual_dtype == expected_dtype
            if not ok:
                issues.append(
                    f"SCHEMA: column '{col}' dtype {actual_dtype!r}, "
                    f"expected {expected_dtype!r}"
                )
            schema_rows.append(
                f"| {col} | {expected_dtype} | "
                f"{'[OK] ' + actual_dtype if ok else '[FAIL] ' + actual_dtype} |"
            )

    # -----------------------------------------------------------------------
    # 2. DATA QUALITY CHECK
    # -----------------------------------------------------------------------
    quality_rows: list = []
    for col in df.columns:
        rate = float(df[col].isna().mean())
        ok = rate <= missing_threshold
        if not ok:
            issues.append(
                f"QUALITY: '{col}' has {rate:.1%} missing "
                f"(threshold {missing_threshold:.0%})"
            )
        quality_rows.append(
            f"| {col} | {rate:.1%} | "
            f"{'[OK]' if ok else '[EXCEEDS THRESHOLD]'} |"
        )

    # -----------------------------------------------------------------------
    # 3. DRIFT DETECTION
    # -----------------------------------------------------------------------
    drift_rows: list = []
    for col in EXPECTED_COLUMNS:
        if col not in df.columns or col not in ref["columns"]:
            continue

        ref_col = ref["columns"][col]
        ref_values = ref_col["values"]
        new_values = df[col].dropna().tolist()

        # KS test -- scipy returns numpy scalars; cast to plain Python for JSON.
        ks_stat_raw, p_value_raw = scipy_stats.ks_2samp(ref_values, new_values)
        ks_stat = float(ks_stat_raw)
        p_value = float(p_value_raw)

        # Mean-shift check
        ref_mean = float(ref_col["mean"])
        ref_std = float(ref_col["std"])
        new_mean = float(df[col].mean())
        shift_stds = abs(new_mean - ref_mean) / ref_std if ref_std > 0 else 0.0

        ks_drift = bool(p_value < ks_pvalue_threshold)
        mean_drift = bool(shift_stds > mean_shift_threshold)
        col_drifted = ks_drift or mean_drift

        if ks_drift:
            issues.append(
                f"DRIFT: '{col}' KS p={p_value:.4f} < {ks_pvalue_threshold} "
                f"(stat={ks_stat:.4f})"
            )
        if mean_drift:
            issues.append(
                f"DRIFT: '{col}' mean shifted {shift_stds:.2f} std devs "
                f"(ref={ref_mean:.4f} -> new={new_mean:.4f})"
            )

        col_results[col] = {
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(p_value, 4),
            "ks_drifted": ks_drift,
            "ref_mean": round(ref_mean, 4),
            "new_mean": round(new_mean, 4),
            "mean_shift_stds": round(shift_stds, 4),
            "mean_drifted": mean_drift,
            "drifted": col_drifted,
        }

        status = "[DRIFT]" if col_drifted else "[OK]"
        drift_rows.append(
            f"| {col} | {ks_stat:.4f} | {p_value:.4f} | "
            f"{ref_mean:.4f} | {new_mean:.4f} | {shift_stds:.2f} std | {status} |"
        )

    # -----------------------------------------------------------------------
    # Assemble report
    # -----------------------------------------------------------------------
    validation_passed = len(issues) == 0
    status_badge = "PASSED" if validation_passed else f"FAILED ({len(issues)} issues)"

    report_json = {
        "validation_passed": validation_passed,
        "n_issues": len(issues),
        "issues": issues,
        "data_shape": {"rows": len(df), "cols": len(df.columns)},
        "column_drift": col_results,
    }

    md_lines = [
        "# Data Validation Report",
        "",
        f"**Status**: {status_badge}  ",
        f"**Rows**: {len(df)}  |  **Columns**: {len(df.columns)}  ",
        f"**Reference rows**: {ref.get('n_rows', '?')}",
        "",
    ]

    if issues:
        md_lines += ["## Issues Found", ""]
        for issue in issues:
            md_lines.append(f"- {issue}")
        md_lines.append("")

    md_lines += [
        "## 1 - Schema Check",
        "",
        "| Column | Expected dtype | Actual |",
        "|--------|---------------|--------|",
        *schema_rows,
        "",
        "## 2 - Data Quality (missing values)",
        "",
        f"Threshold: **{missing_threshold:.0%}**",
        "",
        "| Column | Missing Rate | Status |",
        "|--------|-------------|--------|",
        *quality_rows,
        "",
        "## 3 - Drift Detection",
        "",
        f"KS p-value threshold: **{ks_pvalue_threshold}**  ",
        f"Mean-shift threshold: **{mean_shift_threshold} std devs**",
        "",
        "| Column | KS stat | p-value | Ref mean | New mean | Shift | Status |",
        "|--------|---------|---------|----------|----------|-------|--------|",
        *drift_rows,
        "",
        "## Full JSON",
        "",
        "```json",
        json.dumps(report_json, indent=2),
        "```",
    ]

    md_content = "\n".join(md_lines)
    with open(report.path, "w") as fh:
        fh.write(md_content)

    # Also print to pod stdout (visible in KFP UI logs)
    print("=" * 60)
    print(f"VALIDATION: {status_badge}")
    if issues:
        for issue in issues:
            print(f"  [x] {issue}")
    print("\nColumn drift summary:")
    for col, r in col_results.items():
        tag = "DRIFT" if r["drifted"] else "OK   "
        print(f"  {col:10s} [{tag}]  KS={r['ks_statistic']:.3f}  "
              f"p={r['ks_pvalue']:.3f}  shift={r['mean_shift_stds']:.2f} std")
    print("=" * 60)

    return validation_passed
