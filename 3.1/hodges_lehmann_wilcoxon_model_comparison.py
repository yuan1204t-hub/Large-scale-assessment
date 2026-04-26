# -*- coding: utf-8 -*-
"""
Purpose:
1. Read a wide-format model performance summary table.
2. Perform pairwise paired comparisons for R2, RMSE, and MAE across multiple models.
3. For each pairwise comparison, calculate:
   - Wilcoxon signed-rank test p-value
   - Holm-adjusted p-value
   - Hodges-Lehmann paired difference estimate
   - 95% confidence interval of the Hodges-Lehmann estimate
   - Approximate equivalence judgment based on a predefined delta threshold
4. Generate pairwise comparison tables for each metric.
5. Generate Hodges-Lehmann estimate matrices and Holm-adjusted p-value matrices.
6. Save all results into one Excel file.

Applicable scenarios:
- The same group of datasets has been evaluated by multiple models.
- Each dataset has paired performance metrics from different models.
- The user wants to compare not only statistical significance, but also
  the direction, magnitude, confidence interval, and approximate equivalence
  of model differences.
- The outputs will be used for Chapter 3.1 model comparison analysis.
"""

import os
import math
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, norm


# =========================================================
# 1. Path settings
# =========================================================
# Please enter the path of the wide-format model performance summary table.
input_file = r"Please enter your path here"

# Please enter the output Excel file path.
output_file = r"Please enter your path here"

# Sheet index or sheet name of the input Excel file.
sheet_name = 0


# =========================================================
# 2. Model, metric, and statistical settings
# =========================================================
alpha = 0.05

models = ["M0", "M1", "M2", "GPR", "PLS", "Ridge", "SVM"]

metrics = ["R2", "RMSE", "MAE"]

metric_columns = {
    metric: [f"{model}_{metric}" for model in models]
    for metric in metrics
}

# Approximate equivalence thresholds.
# These values can be adjusted according to the research context.
delta_dict = {
    "R2": 0.01,
    "RMSE": 0.05,
    "MAE": 0.05
}


# =========================================================
# 3. Utility functions
# =========================================================
def keep_existing_columns(df, columns):
    """Keep only columns that actually exist in the dataframe."""
    return [col for col in columns if col in df.columns]


def get_model_name(column_name):
    """Extract model name from a metric column name."""
    return str(column_name).split("_")[0]


def holm_correction(p_values_dict):
    """
    Apply Holm correction to a dictionary of p-values.

    Parameters
    ----------
    p_values_dict : dict
        Dictionary in the form {comparison_name: raw_p_value}.

    Returns
    -------
    dict
        Dictionary in the form {comparison_name: holm_adjusted_p_value}.
    """

    valid_items = [
        (name, p)
        for name, p in p_values_dict.items()
        if not pd.isna(p)
    ]

    invalid_items = [
        (name, p)
        for name, p in p_values_dict.items()
        if pd.isna(p)
    ]

    valid_items = sorted(valid_items, key=lambda x: x[1])
    m = len(valid_items)

    adjusted = {}
    previous_adjusted_p = 0.0

    for i, (name, p_value) in enumerate(valid_items):
        adjusted_p = (m - i) * p_value
        adjusted_p = max(adjusted_p, previous_adjusted_p)
        adjusted_p = min(adjusted_p, 1.0)

        adjusted[name] = adjusted_p
        previous_adjusted_p = adjusted_p

    for name, _ in invalid_items:
        adjusted[name] = np.nan

    return adjusted


def hodges_lehmann_paired(diff):
    """
    Calculate the Hodges-Lehmann paired difference estimate.

    For paired data, the Hodges-Lehmann estimate is the median of Walsh averages.
    """

    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]

    n = len(diff)

    if n == 0:
        return np.nan, np.array([])

    walsh_averages = []

    for i in range(n):
        for j in range(i, n):
            walsh_averages.append((diff[i] + diff[j]) / 2.0)

    walsh_averages = np.sort(np.array(walsh_averages))
    hl_estimate = np.median(walsh_averages)

    return hl_estimate, walsh_averages


def hodges_lehmann_ci_paired(diff, alpha=0.05):
    """
    Calculate an approximate confidence interval for the Hodges-Lehmann estimate.

    The confidence interval is based on the ordered Walsh averages.
    """

    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]

    # Zero differences are removed to match the Wilcoxon signed-rank setting.
    diff = diff[diff != 0]

    n = len(diff)

    if n == 0:
        return np.nan, np.nan, np.nan, 0

    hl_estimate, walsh_averages = hodges_lehmann_paired(diff)

    number_of_walsh = len(walsh_averages)

    z_value = norm.ppf(1 - alpha / 2)

    mean_w = n * (n + 1) / 4
    var_w = n * (n + 1) * (2 * n + 1) / 24
    se_w = math.sqrt(var_w)

    c_alpha = math.floor(mean_w - z_value * se_w)

    lower_index = int(max(0, c_alpha))
    upper_index = int(number_of_walsh - lower_index - 1)

    lower_index = max(0, min(lower_index, number_of_walsh - 1))
    upper_index = max(0, min(upper_index, number_of_walsh - 1))

    ci_low = walsh_averages[lower_index]
    ci_high = walsh_averages[upper_index]

    return hl_estimate, ci_low, ci_high, n


def significance_stars(p_value):
    """Convert a p-value into significance stars."""

    if pd.isna(p_value):
        return ""

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return ""


def run_pairwise_hl_analysis(df, model_columns, metric_name, delta, alpha=0.05):
    """
    Run pairwise Wilcoxon tests and Hodges-Lehmann analyses for one metric.

    Parameters
    ----------
    df : pandas.DataFrame
        Input wide-format dataframe.

    model_columns : dict
        Dictionary in the form {model_name: column_name}.

    metric_name : str
        Name of the metric, such as R2, RMSE, or MAE.

    delta : float
        Threshold for approximate equivalence.

    alpha : float
        Significance level used for confidence interval calculation.

    Returns
    -------
    pandas.DataFrame
        Pairwise comparison results.
    """

    results = []
    raw_p_values = {}

    model_names = list(model_columns.keys())

    for model_a, model_b in itertools.combinations(model_names, 2):
        col_a = model_columns[model_a]
        col_b = model_columns[model_b]

        pair_df = df[[col_a, col_b]].dropna(axis=0, how="any").copy()

        x = pd.to_numeric(pair_df[col_a], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(pair_df[col_b], errors="coerce").to_numpy(dtype=float)

        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x = x[valid_mask]
        y = y[valid_mask]

        diff = x - y

        comparison_name = f"{model_a} vs {model_b}"
        direction = f"{model_a} - {model_b}"

        if len(diff) == 0:
            wilcoxon_stat = np.nan
            raw_p = np.nan
        elif np.allclose(diff, 0, equal_nan=True):
            wilcoxon_stat = 0.0
            raw_p = 1.0
        else:
            try:
                wilcoxon_result = wilcoxon(
                    x,
                    y,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto"
                )
                wilcoxon_stat = wilcoxon_result.statistic
                raw_p = wilcoxon_result.pvalue
            except ValueError:
                wilcoxon_stat = np.nan
                raw_p = np.nan

        raw_p_values[comparison_name] = raw_p

        hl_estimate, ci_low, ci_high, n_nonzero = hodges_lehmann_ci_paired(
            diff,
            alpha=alpha
        )

        approximately_equivalent = (
            pd.notna(ci_low)
            and pd.notna(ci_high)
            and ci_low >= -delta
            and ci_high <= delta
        )

        results.append({
            "Metric": metric_name,
            "Comparison": comparison_name,
            "Direction": direction,
            "Model_A": model_a,
            "Model_B": model_b,
            "N_total": len(diff),
            "N_nonzero_diff": n_nonzero,
            "Wilcoxon_stat": wilcoxon_stat,
            "Wilcoxon_p": raw_p,
            "HL_estimate": hl_estimate,
            "HL_CI_low": ci_low,
            "HL_CI_high": ci_high,
            "Delta": delta,
            "Equivalent_by_CI": "Yes" if approximately_equivalent else "No"
        })

    holm_p_values = holm_correction(raw_p_values)

    for row in results:
        row["Holm_p"] = holm_p_values[row["Comparison"]]
        row["Sig_raw"] = significance_stars(row["Wilcoxon_p"])
        row["Sig_Holm"] = significance_stars(row["Holm_p"])

    result_df = pd.DataFrame(results)

    result_df = result_df[
        [
            "Metric",
            "Comparison",
            "Direction",
            "Model_A",
            "Model_B",
            "N_total",
            "N_nonzero_diff",
            "Wilcoxon_stat",
            "Wilcoxon_p",
            "Sig_raw",
            "Holm_p",
            "Sig_Holm",
            "HL_estimate",
            "HL_CI_low",
            "HL_CI_high",
            "Delta",
            "Equivalent_by_CI"
        ]
    ]

    return result_df


def build_holm_p_matrix(result_df, model_names):
    """
    Build a symmetric matrix of Holm-adjusted p-values.

    The value in each cell represents the Holm-adjusted p-value for the
    comparison between the row model and the column model.
    """

    matrix = pd.DataFrame(np.nan, index=model_names, columns=model_names)

    for _, row in result_df.iterrows():
        model_a = row["Model_A"]
        model_b = row["Model_B"]
        p_value = row["Holm_p"]

        matrix.loc[model_a, model_b] = p_value
        matrix.loc[model_b, model_a] = p_value

    for model in model_names:
        matrix.loc[model, model] = 0.0

    return matrix


def build_hl_matrix(result_df, model_names):
    """
    Build a directional matrix of Hodges-Lehmann estimates.

    Cell meaning:
    row model - column model.
    """

    matrix = pd.DataFrame(np.nan, index=model_names, columns=model_names)

    for _, row in result_df.iterrows():
        model_a = row["Model_A"]
        model_b = row["Model_B"]
        hl_estimate = row["HL_estimate"]

        matrix.loc[model_a, model_b] = hl_estimate
        matrix.loc[model_b, model_a] = -hl_estimate

    for model in model_names:
        matrix.loc[model, model] = 0.0

    return matrix


# =========================================================
# 4. Main program
# =========================================================
def main():
    """Run Hodges-Lehmann and Wilcoxon pairwise model comparison."""

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = pd.read_excel(input_file, sheet_name=sheet_name)

    print("[INFO] Data loaded successfully.")
    print(f"[INFO] Data shape: {df.shape}")
    print("[INFO] Columns:")
    print(df.columns.tolist())

    all_pairwise_results = []

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for metric_name in metrics:
            print(f"\n[INFO] Processing metric: {metric_name}")

            expected_columns = metric_columns[metric_name]
            existing_columns = keep_existing_columns(df, expected_columns)

            if len(existing_columns) < 2:
                print(f"[WARNING] Fewer than two columns found for {metric_name}. Skipped.")
                continue

            model_columns = {
                get_model_name(col): col
                for col in existing_columns
            }

            available_model_names = list(model_columns.keys())

            missing_columns = [
                col for col in expected_columns
                if col not in existing_columns
            ]

            if missing_columns:
                print(f"[WARNING] Missing columns for {metric_name}: {missing_columns}")

            delta = delta_dict[metric_name]

            pairwise_df = run_pairwise_hl_analysis(
                df=df,
                model_columns=model_columns,
                metric_name=metric_name,
                delta=delta,
                alpha=alpha
            )

            all_pairwise_results.append(pairwise_df)

            hl_matrix = build_hl_matrix(
                result_df=pairwise_df,
                model_names=available_model_names
            )

            holm_p_matrix = build_holm_p_matrix(
                result_df=pairwise_df,
                model_names=available_model_names
            )

            pairwise_df.to_excel(
                writer,
                sheet_name=f"{metric_name}_pairwise",
                index=False
            )

            hl_matrix.to_excel(
                writer,
                sheet_name=f"{metric_name}_HL_matrix"
            )

            holm_p_matrix.to_excel(
                writer,
                sheet_name=f"{metric_name}_Holm_p_matrix"
            )

        if all_pairwise_results:
            summary_df = pd.concat(all_pairwise_results, ignore_index=True)
            summary_df.to_excel(writer, sheet_name="All_Summary", index=False)
        else:
            empty_df = pd.DataFrame()
            empty_df.to_excel(writer, sheet_name="All_Summary", index=False)

    print("\n[INFO] Hodges-Lehmann and Wilcoxon analyses completed.")
    print(f"[INFO] Output saved to: {output_file}")


if __name__ == "__main__":
    main()