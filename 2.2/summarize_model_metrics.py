# -*- coding: utf-8 -*-
"""
Purpose:
1. Read a wide-format model performance summary table.
2. Calculate descriptive statistics for R2, RMSE, and MAE across different models.
3. Identify the best-performing model for each dataset under each metric.
4. Count how many times each model performs best.
5. Convert the wide-format table into a long-format table for later plotting
   and statistical analysis.
6. Save all results as Excel files.

Applicable scenarios:
- A model performance summary table has already been generated.
- The table contains R2, RMSE, and MAE columns for multiple models.
- The user wants to summarize model performance before formal statistical tests.
- The outputs will be used for Chapter 3.1 descriptive analysis, visualization,
  and model comparison.
"""

import os
import numpy as np
import pandas as pd


# =========================================================
# 1. Path settings
# =========================================================
# Please enter the path of the wide-format model performance summary table.
input_file = r"Please enter your path here"

# Please enter the output folder path.
output_dir = r"Please enter your path here"


# =========================================================
# 2. Model and metric settings
# =========================================================
models = ["M0", "M1", "M2", "GPR", "PLS", "Ridge", "SVM"]

metric_columns = {
    "R2": [f"{model}_R2" for model in models],
    "RMSE": [f"{model}_RMSE" for model in models],
    "MAE": [f"{model}_MAE" for model in models],
}

# For R2, a larger value indicates better performance.
# For RMSE and MAE, a smaller value indicates better performance.
best_direction = {
    "R2": "max",
    "RMSE": "min",
    "MAE": "min",
}


# =========================================================
# 3. Utility functions
# =========================================================
def get_model_name(column_name):
    """Extract model name from a metric column name."""
    return str(column_name).split("_")[0]


def calculate_descriptive_statistics(df, columns, metric_name):
    """
    Calculate descriptive statistics for a group of metric columns.

    The output includes:
    - n
    - mean
    - standard deviation
    - median
    - Q1
    - Q3
    - IQR
    - minimum
    - maximum
    """

    rows = []

    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(values) == 0:
            rows.append({
                "Metric": metric_name,
                "Model": get_model_name(col),
                "Column": col,
                "n": 0,
                "Mean": np.nan,
                "SD": np.nan,
                "Median": np.nan,
                "Q1": np.nan,
                "Q3": np.nan,
                "IQR": np.nan,
                "Min": np.nan,
                "Max": np.nan
            })
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        rows.append({
            "Metric": metric_name,
            "Model": get_model_name(col),
            "Column": col,
            "n": len(values),
            "Mean": values.mean(),
            "SD": values.std(),
            "Median": values.median(),
            "Q1": q1,
            "Q3": q3,
            "IQR": q3 - q1,
            "Min": values.min(),
            "Max": values.max()
        })

    return pd.DataFrame(rows)


def find_best_column(row, columns, direction):
    """
    Find the best-performing model column for one dataset.

    For R2:
    - direction = "max"

    For RMSE and MAE:
    - direction = "min"
    """

    values = pd.to_numeric(row[columns], errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    if direction == "max":
        return values.idxmax()

    if direction == "min":
        return values.idxmin()

    raise ValueError("direction must be either 'max' or 'min'.")


def calculate_best_model_counts(df, columns, metric_name, direction):
    """
    Count how many times each model is the best-performing model.

    Ties are handled by pandas idxmax or idxmin, which returns the first
    occurrence among tied values.
    """

    best_col_name = f"Best_{metric_name}"

    temp = df.copy()
    temp[best_col_name] = temp.apply(
        lambda row: find_best_column(row, columns, direction),
        axis=1
    )

    count_df = temp[best_col_name].value_counts(dropna=False).reset_index()
    count_df.columns = ["Best_Column", "Count"]

    count_df["Metric"] = metric_name
    count_df["Model"] = count_df["Best_Column"].apply(
        lambda x: get_model_name(x) if pd.notna(x) else np.nan
    )

    total_n = count_df["Count"].sum()
    count_df["Percentage"] = count_df["Count"] / total_n * 100

    count_df = count_df[
        ["Metric", "Model", "Best_Column", "Count", "Percentage"]
    ]

    return count_df


def convert_to_long_format(df, columns, metric_name):
    """
    Convert selected wide-format metric columns into long format.

    Long-format data are useful for:
    - violin plots
    - box plots
    - grouped summaries
    - statistical tests
    """

    temp = df[columns].copy()

    long_df = temp.melt(
        var_name="Column",
        value_name="Value"
    )

    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df = long_df.dropna(subset=["Value"]).reset_index(drop=True)

    long_df["Metric"] = metric_name
    long_df["Model"] = long_df["Column"].apply(get_model_name)

    return long_df[["Metric", "Model", "Column", "Value"]]


def keep_existing_columns(df, columns):
    """Keep only columns that actually exist in the dataframe."""
    return [col for col in columns if col in df.columns]


# =========================================================
# 4. Main program
# =========================================================
def main():
    """Run descriptive statistics and best-model counting."""

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_excel(input_file)

    print("[INFO] Data loaded successfully.")
    print(f"[INFO] Data shape: {df.shape}")

    descriptive_tables = []
    best_count_tables = []
    long_tables = []

    for metric_name, columns in metric_columns.items():
        existing_cols = keep_existing_columns(df, columns)

        if len(existing_cols) == 0:
            print(f"[WARNING] No columns found for {metric_name}. Skipped.")
            continue

        print(f"[INFO] Processing metric: {metric_name}")
        print(f"[INFO] Available columns: {existing_cols}")

        desc_df = calculate_descriptive_statistics(
            df=df,
            columns=existing_cols,
            metric_name=metric_name
        )

        best_df = calculate_best_model_counts(
            df=df,
            columns=existing_cols,
            metric_name=metric_name,
            direction=best_direction[metric_name]
        )

        long_df = convert_to_long_format(
            df=df,
            columns=existing_cols,
            metric_name=metric_name
        )

        descriptive_tables.append(desc_df)
        best_count_tables.append(best_df)
        long_tables.append(long_df)

    if not descriptive_tables:
        print("[ERROR] No valid metric columns were found. Please check column names.")
        return

    descriptive_all = pd.concat(descriptive_tables, ignore_index=True)
    best_counts_all = pd.concat(best_count_tables, ignore_index=True)
    long_all = pd.concat(long_tables, ignore_index=True)

    # =====================================================
    # 5. Save outputs
    # =====================================================
    descriptive_path = os.path.join(output_dir, "descriptive_statistics.xlsx")
    best_counts_path = os.path.join(output_dir, "best_model_counts.xlsx")
    long_format_path = os.path.join(output_dir, "long_format_metrics.xlsx")

    descriptive_all.to_excel(descriptive_path, index=False)
    best_counts_all.to_excel(best_counts_path, index=False)
    long_all.to_excel(long_format_path, index=False)

    print("\n[INFO] Descriptive statistics and best-model counting completed.")
    print(f"[INFO] Descriptive statistics saved to: {descriptive_path}")
    print(f"[INFO] Best-model counts saved to: {best_counts_path}")
    print(f"[INFO] Long-format data saved to: {long_format_path}")


if __name__ == "__main__":
    main()