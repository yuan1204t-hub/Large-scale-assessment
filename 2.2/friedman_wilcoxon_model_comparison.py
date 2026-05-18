# -*- coding: utf-8 -*-
"""
Purpose:
1. Read a wide-format model performance summary table.
2. Perform Friedman overall tests for R2, RMSE, and MAE across multiple models.
3. Perform Wilcoxon signed-rank pairwise comparisons between models.
4. Apply Bonferroni correction to pairwise p-values.
5. Save overall test results, pairwise comparison results, complete-case sample
   sizes, and long-format complete-case data.

Applicable scenarios:
- The same group of datasets has been evaluated by multiple models.
- Each dataset has paired performance metrics from different models.
- The analysis follows a repeated-measures or paired-comparison design.
- The user wants to test whether model performance differs significantly
  across R2, RMSE, and MAE.
"""

import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


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


# =========================================================
# 3. Utility functions
# =========================================================
def get_model_name(column_name):
    """Extract model name from a metric column name."""
    return str(column_name).split("_")[0]


def keep_existing_columns(df, columns):
    """Keep only columns that actually exist in the dataframe."""
    return [col for col in columns if col in df.columns]


def bonferroni_correction(p_values):
    """
    Apply Bonferroni correction to a list of p-values.

    Corrected p-value = raw p-value * number of comparisons.
    The corrected p-value is capped at 1.0.
    """

    p_values = np.array(p_values, dtype=float)
    m = len(p_values)

    return np.minimum(p_values * m, 1.0)


def significance_label(p_value):
    """Convert a p-value into a significance label."""

    if pd.isna(p_value):
        return ""

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return "ns"


def run_friedman_test(data, columns, metric_name):
    """
    Run the Friedman overall test for one metric.

    Friedman test requires complete cases. Therefore, only datasets with
    non-missing values for all included models are used.
    """

    sub = data[columns].apply(pd.to_numeric, errors="coerce")
    sub_complete = sub.dropna(axis=0, how="any").copy()

    n_complete = len(sub_complete)
    model_names = [get_model_name(col) for col in columns]

    if n_complete == 0 or len(columns) < 3:
        return {
            "Metric": metric_name,
            "Models": ", ".join(model_names),
            "Model_Count": len(columns),
            "Complete_Case_n": n_complete,
            "Friedman_Statistic": np.nan,
            "p_value": np.nan,
            "Significance": ""
        }, sub_complete

    arrays = [sub_complete[col].values for col in columns]
    statistic, p_value = friedmanchisquare(*arrays)

    return {
        "Metric": metric_name,
        "Models": ", ".join(model_names),
        "Model_Count": len(columns),
        "Complete_Case_n": n_complete,
        "Friedman_Statistic": statistic,
        "p_value": p_value,
        "Significance": significance_label(p_value)
    }, sub_complete


def run_wilcoxon_pairwise_tests(data, columns, metric_name):
    """
    Run Wilcoxon signed-rank pairwise comparisons for one metric.

    Each pair is tested using paired non-missing observations only.
    Bonferroni correction is applied within the metric.
    """

    pair_rows = []
    raw_p_values = []

    numeric_data = data[columns].apply(pd.to_numeric, errors="coerce")
    pairs = list(itertools.combinations(columns, 2))

    for col_1, col_2 in pairs:
        pair_df = numeric_data[[col_1, col_2]].dropna(axis=0, how="any").copy()
        n_pair = len(pair_df)

        model_1 = get_model_name(col_1)
        model_2 = get_model_name(col_2)

        if n_pair == 0:
            statistic = np.nan
            raw_p = np.nan
        else:
            x = pair_df[col_1].values
            y = pair_df[col_2].values

            if np.allclose(x, y, equal_nan=True):
                statistic = 0.0
                raw_p = 1.0
            else:
                try:
                    statistic, raw_p = wilcoxon(
                        x,
                        y,
                        zero_method="wilcox",
                        alternative="two-sided"
                    )
                except ValueError:
                    statistic = np.nan
                    raw_p = np.nan

        raw_p_values.append(raw_p)

        pair_rows.append({
            "Metric": metric_name,
            "Model_1": model_1,
            "Model_2": model_2,
            "Column_1": col_1,
            "Column_2": col_2,
            "n": n_pair,
            "Wilcoxon_Statistic": statistic,
            "Raw_p": raw_p
        })

    corrected_p_values = bonferroni_correction(raw_p_values)

    for row, corrected_p in zip(pair_rows, corrected_p_values):
        row["Bonferroni_p"] = corrected_p
        row["Significance"] = significance_label(corrected_p)

    return pd.DataFrame(pair_rows)


def make_complete_case_long_table(sub_complete, metric_name):
    """
    Convert complete-case metric data into long format.

    The output can be used for plotting or checking the data used in the
    Friedman test.
    """

    if sub_complete.empty:
        return pd.DataFrame(
            columns=["Dataset_Index", "Column", "Value", "Metric", "Model"]
        )

    temp = sub_complete.copy()
    temp["Dataset_Index"] = np.arange(len(temp))

    long_df = temp.melt(
        id_vars="Dataset_Index",
        var_name="Column",
        value_name="Value"
    )

    long_df["Metric"] = metric_name
    long_df["Model"] = long_df["Column"].apply(get_model_name)

    return long_df[["Dataset_Index", "Metric", "Model", "Column", "Value"]]


# =========================================================
# 4. Main program
# =========================================================
def main():
    """Run Friedman overall tests and Wilcoxon pairwise comparisons."""

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_excel(input_file)

    print("[INFO] Data loaded successfully.")
    print(f"[INFO] Data shape: {df.shape}")

    friedman_results = []
    pairwise_results = []
    sample_size_summary = []
    long_complete_tables = []

    for metric_name, columns in metric_columns.items():
        existing_cols = keep_existing_columns(df, columns)

        if len(existing_cols) < 2:
            print(f"[WARNING] Fewer than two columns found for {metric_name}. Skipped.")
            continue

        print(f"[INFO] Processing metric: {metric_name}")
        print(f"[INFO] Available columns: {existing_cols}")

        friedman_row, sub_complete = run_friedman_test(
            data=df,
            columns=existing_cols,
            metric_name=metric_name
        )

        friedman_results.append(friedman_row)

        sample_size_summary.append({
            "Metric": metric_name,
            "Model_Count": len(existing_cols),
            "Complete_Case_n": friedman_row["Complete_Case_n"]
        })

        if len(existing_cols) >= 2:
            pairwise_df = run_wilcoxon_pairwise_tests(
                data=df,
                columns=existing_cols,
                metric_name=metric_name
            )

            pairwise_results.append(pairwise_df)

        long_df = make_complete_case_long_table(
            sub_complete=sub_complete,
            metric_name=metric_name
        )

        long_complete_tables.append(long_df)

    if not friedman_results:
        print("[ERROR] No valid metric columns were found. Please check column names.")
        return

    friedman_df = pd.DataFrame(friedman_results)

    if pairwise_results:
        pairwise_df = pd.concat(pairwise_results, ignore_index=True)
    else:
        pairwise_df = pd.DataFrame()

    sample_size_df = pd.DataFrame(sample_size_summary)

    if long_complete_tables:
        long_complete_df = pd.concat(long_complete_tables, ignore_index=True)
    else:
        long_complete_df = pd.DataFrame()

    # =====================================================
    # 5. Save outputs
    # =====================================================
    friedman_path = os.path.join(output_dir, "friedman_overall_tests.xlsx")
    pairwise_path = os.path.join(output_dir, "wilcoxon_pairwise_bonferroni.xlsx")
    sample_size_path = os.path.join(output_dir, "complete_case_sample_sizes.xlsx")
    long_path = os.path.join(output_dir, "complete_case_long_format.xlsx")

    friedman_df.to_excel(friedman_path, index=False)
    pairwise_df.to_excel(pairwise_path, index=False)
    sample_size_df.to_excel(sample_size_path, index=False)
    long_complete_df.to_excel(long_path, index=False)

    print("\n[INFO] Friedman and Wilcoxon analyses completed.")
    print(f"[INFO] Friedman overall tests saved to: {friedman_path}")
    print(f"[INFO] Wilcoxon pairwise comparisons saved to: {pairwise_path}")
    print(f"[INFO] Complete-case sample sizes saved to: {sample_size_path}")
    print(f"[INFO] Complete-case long-format data saved to: {long_path}")


if __name__ == "__main__":
    main()