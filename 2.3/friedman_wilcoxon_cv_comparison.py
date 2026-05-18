# -*- coding: utf-8 -*-
"""
Purpose:
1. Read a wide-format cross-validation metric summary table.
2. Perform Friedman overall tests for RMSECV, MAECV, and Q2 across multiple models.
3. Perform Wilcoxon signed-rank pairwise comparisons between models.
4. Apply Holm correction to pairwise p-values.
5. Save overall test results, pairwise comparison results, complete-case sample
   sizes, and complete-case long-format data.

Applicable scenarios:
- The same group of datasets has been evaluated by multiple models.
- Each dataset has paired cross-validation performance metrics from different models.
- The tested metrics include RMSECV, MAECV, and Q2.
- The analysis follows a repeated-measures or paired-comparison design.
- The outputs will be used for Chapter 3.2 predictive performance comparison.
"""

import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


# =========================================================
# 1. Path settings
# =========================================================
# Please enter the path of the wide-format cross-validation metric summary table.
input_file = r"Please enter your path here"

# Please enter the output folder path.
output_dir = r"Please enter your path here"


# =========================================================
# 2. Model and metric settings
# =========================================================
models = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]

metrics = ["RMSECV", "MAECV", "Q2"]

metric_columns = {
    metric: [f"{model}_{metric}" for model in models]
    for metric in metrics
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


def holm_correction(p_values):
    """
    Apply Holm correction to a list of p-values.

    This function ignores missing p-values during correction and then returns
    the adjusted p-values in the original order.
    """

    p_values = np.array(p_values, dtype=float)

    adjusted_p_values = np.full_like(p_values, fill_value=np.nan, dtype=float)

    valid_mask = ~np.isnan(p_values)
    valid_p_values = p_values[valid_mask]

    if len(valid_p_values) == 0:
        return adjusted_p_values

    m = len(valid_p_values)

    sorted_indices = np.argsort(valid_p_values)
    sorted_p_values = valid_p_values[sorted_indices]

    sorted_adjusted = np.zeros(m)
    previous_adjusted = 0.0

    for i in range(m):
        adjusted = sorted_p_values[i] * (m - i)
        adjusted = max(adjusted, previous_adjusted)
        adjusted = min(adjusted, 1.0)

        sorted_adjusted[i] = adjusted
        previous_adjusted = adjusted

    valid_adjusted = np.zeros(m)
    valid_adjusted[sorted_indices] = sorted_adjusted

    adjusted_p_values[valid_mask] = valid_adjusted

    return adjusted_p_values


def run_friedman_test(df, columns, metric_name):
    """
    Run the Friedman overall test for one metric.

    Friedman test requires complete cases. Therefore, only rows with no missing
    values across all included model columns are used.
    """

    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")
    complete_df = numeric_df.dropna(axis=0, how="any").copy()

    n_complete = len(complete_df)
    model_names = [get_model_name(col) for col in columns]

    if n_complete == 0 or len(columns) < 3:
        return {
            "Metric": metric_name,
            "Models": ", ".join(model_names),
            "Model_Count": len(columns),
            "Complete_Case_n": n_complete,
            "Friedman_Statistic": np.nan,
            "Degrees_of_Freedom": np.nan,
            "p_value": np.nan,
            "Significance": ""
        }, complete_df

    arrays = [complete_df[col].values for col in columns]
    statistic, p_value = friedmanchisquare(*arrays)

    return {
        "Metric": metric_name,
        "Models": ", ".join(model_names),
        "Model_Count": len(columns),
        "Complete_Case_n": n_complete,
        "Friedman_Statistic": statistic,
        "Degrees_of_Freedom": len(columns) - 1,
        "p_value": p_value,
        "Significance": significance_label(p_value)
    }, complete_df


def run_wilcoxon_pairwise_tests(df, columns, metric_name):
    """
    Run Wilcoxon signed-rank pairwise comparisons for one metric.

    Each pair is tested using paired non-missing observations only.
    Holm correction is applied within each metric.
    """

    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")

    pair_rows = []
    raw_p_values = []

    pairs = list(itertools.combinations(columns, 2))

    for col_1, col_2 in pairs:
        pair_df = numeric_df[[col_1, col_2]].dropna(axis=0, how="any").copy()

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
                        alternative="two-sided",
                        method="auto"
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

    adjusted_p_values = holm_correction(raw_p_values)

    for row, adjusted_p in zip(pair_rows, adjusted_p_values):
        row["Holm_p"] = adjusted_p
        row["Significance"] = significance_label(adjusted_p)

    return pd.DataFrame(pair_rows)


def make_complete_case_long_table(complete_df, metric_name):
    """
    Convert complete-case metric data into long format.

    This table records the actual data used in the Friedman test.
    """

    if complete_df.empty:
        return pd.DataFrame(
            columns=["Dataset_Index", "Metric", "Model", "Column", "Value"]
        )

    temp = complete_df.copy()
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
    complete_case_long_tables = []
    missing_column_records = []

    for metric_name in metrics:
        expected_columns = metric_columns[metric_name]
        existing_columns = keep_existing_columns(df, expected_columns)

        missing_columns = [
            col for col in expected_columns
            if col not in existing_columns
        ]

        for col in missing_columns:
            missing_column_records.append({
                "Metric": metric_name,
                "Missing_Column": col
            })

        if len(existing_columns) < 2:
            print(f"[WARNING] Fewer than two columns found for {metric_name}. Skipped.")
            continue

        print(f"[INFO] Processing metric: {metric_name}")
        print(f"[INFO] Available columns: {existing_columns}")

        friedman_row, complete_df = run_friedman_test(
            df=df,
            columns=existing_columns,
            metric_name=metric_name
        )

        friedman_results.append(friedman_row)

        sample_size_summary.append({
            "Metric": metric_name,
            "Model_Count": len(existing_columns),
            "Complete_Case_n": friedman_row["Complete_Case_n"]
        })

        pairwise_df = run_wilcoxon_pairwise_tests(
            df=df,
            columns=existing_columns,
            metric_name=metric_name
        )

        pairwise_results.append(pairwise_df)

        long_df = make_complete_case_long_table(
            complete_df=complete_df,
            metric_name=metric_name
        )

        complete_case_long_tables.append(long_df)

    if not friedman_results:
        print("[ERROR] No valid metric columns were found. Please check column names.")
        return

    friedman_df = pd.DataFrame(friedman_results)

    if pairwise_results:
        pairwise_df = pd.concat(pairwise_results, ignore_index=True)
    else:
        pairwise_df = pd.DataFrame()

    sample_size_df = pd.DataFrame(sample_size_summary)

    if complete_case_long_tables:
        complete_case_long_df = pd.concat(complete_case_long_tables, ignore_index=True)
    else:
        complete_case_long_df = pd.DataFrame()

    missing_columns_df = pd.DataFrame(missing_column_records)

    # =====================================================
    # 5. Save outputs
    # =====================================================
    output_file = os.path.join(output_dir, "friedman_wilcoxon_cv_comparison.xlsx")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        friedman_df.to_excel(writer, sheet_name="Friedman_Overall", index=False)
        pairwise_df.to_excel(writer, sheet_name="Wilcoxon_Holm_Pairwise", index=False)
        sample_size_df.to_excel(writer, sheet_name="Complete_Case_N", index=False)
        complete_case_long_df.to_excel(writer, sheet_name="Complete_Case_Long", index=False)
        missing_columns_df.to_excel(writer, sheet_name="Missing_Columns", index=False)

    print("\n[INFO] Friedman and Wilcoxon analyses completed.")
    print(f"[INFO] Output saved to: {output_file}")


if __name__ == "__main__":
    main()