# -*- coding: utf-8 -*-
"""
Purpose:
1. Read a wide-format cross-validation metric summary table.
2. Perform Shapiro-Wilk normality tests for each model and each metric.
3. The tested metrics include:
   - RMSECV
   - MAECV
   - Q2
4. Output the Shapiro-Wilk W statistic, p-value, sample size, and normality
   judgment for each model-metric combination.
5. Save the normality test results as an Excel file.

Applicable scenarios:
- A cross-validation metric summary table has already been generated.
- The table contains RMSECV, MAECV, and Q2 columns for multiple models.
- The user wants to check whether each model-metric distribution is
  approximately normal.
- The results will be used to support the choice of non-parametric statistical
  tests in Chapter 3.2 predictive performance comparison.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import shapiro


# =========================================================
# 1. Path settings
# =========================================================
# Please enter the path of the wide-format cross-validation metric summary table.
input_file = r"Please enter your path here"

# Please enter the output Excel file path.
output_file = r"Please enter your path here"


# =========================================================
# 2. Model and metric settings
# =========================================================
models = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]

metrics = ["RMSECV", "MAECV", "Q2"]

alpha = 0.05


# =========================================================
# 3. Utility functions
# =========================================================
def significance_label(p_value):
    """Convert a p-value into a simple significance label."""

    if pd.isna(p_value):
        return ""

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return "ns"


def normality_judgment(p_value, alpha=0.05):
    """
    Judge whether the distribution can be regarded as approximately normal.

    In the Shapiro-Wilk test:
    - p_value > alpha suggests no significant departure from normality.
    - p_value <= alpha suggests a significant departure from normality.
    """

    if pd.isna(p_value):
        return "Not tested"

    if p_value > alpha:
        return "Yes"

    return "No"


def run_shapiro_test_for_column(df, column_name, metric, model, alpha=0.05):
    """
    Run the Shapiro-Wilk normality test for one column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    column_name : str
        Column to be tested.

    metric : str
        Metric name, such as RMSECV, MAECV, or Q2.

    model : str
        Model name, such as M0, M1, ..., M6.

    alpha : float
        Significance level for normality judgment.

    Returns
    -------
    dict
        A dictionary containing test results.
    """

    values = pd.to_numeric(df[column_name], errors="coerce").dropna()

    if len(values) < 3:
        return {
            "Metric": metric,
            "Model": model,
            "Column": column_name,
            "n": len(values),
            "W_Statistic": np.nan,
            "p_value": np.nan,
            "Significance": "",
            "Is_Normal": "Not tested",
            "Note": "Fewer than 3 valid observations."
        }

    if values.nunique() < 3:
        return {
            "Metric": metric,
            "Model": model,
            "Column": column_name,
            "n": len(values),
            "W_Statistic": np.nan,
            "p_value": np.nan,
            "Significance": "",
            "Is_Normal": "Not tested",
            "Note": "Fewer than 3 unique values."
        }

    statistic, p_value = shapiro(values)

    return {
        "Metric": metric,
        "Model": model,
        "Column": column_name,
        "n": len(values),
        "W_Statistic": statistic,
        "p_value": p_value,
        "Significance": significance_label(p_value),
        "Is_Normal": normality_judgment(p_value, alpha=alpha),
        "Note": ""
    }


def create_wide_summary(result_df):
    """
    Create a wide-format summary table for easier reporting.

    The output includes W statistics and p-values for each metric.
    """

    if result_df.empty:
        return pd.DataFrame()

    wide_df = result_df.pivot(
        index="Model",
        columns="Metric",
        values=["W_Statistic", "p_value", "Is_Normal"]
    )

    wide_df.columns = [
        f"{value}_{metric}"
        for value, metric in wide_df.columns
    ]

    wide_df = wide_df.reset_index()

    return wide_df


# =========================================================
# 4. Main program
# =========================================================
def main():
    """Run Shapiro-Wilk normality tests for all model-metric combinations."""

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = pd.read_excel(input_file)

    print("[INFO] Data loaded successfully.")
    print(f"[INFO] Data shape: {df.shape}")

    results = []
    missing_columns = []

    for metric in metrics:
        for model in models:
            column_name = f"{model}_{metric}"

            if column_name not in df.columns:
                missing_columns.append(column_name)
                continue

            result = run_shapiro_test_for_column(
                df=df,
                column_name=column_name,
                metric=metric,
                model=model,
                alpha=alpha
            )

            results.append(result)

    result_df = pd.DataFrame(results)
    wide_summary_df = create_wide_summary(result_df)

    missing_df = pd.DataFrame({
        "Missing_Column": missing_columns
    })

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="Shapiro_Long", index=False)
        wide_summary_df.to_excel(writer, sheet_name="Shapiro_Wide", index=False)
        missing_df.to_excel(writer, sheet_name="Missing_Columns", index=False)

    print("\n[INFO] Shapiro-Wilk normality tests completed.")
    print(f"[INFO] Tested columns: {len(result_df)}")
    print(f"[INFO] Missing columns: {len(missing_columns)}")
    print(f"[INFO] Output saved to: {output_file}")


if __name__ == "__main__":
    main()