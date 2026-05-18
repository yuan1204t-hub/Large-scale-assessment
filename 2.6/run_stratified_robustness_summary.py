# -*- coding: utf-8 -*-
"""
Script name
-----------
run_stratified_robustness_summary.py

Purpose
-------
This script summarizes stratified robustness results for a multi-model comparison
study of extraction-process datasets.

Main tasks
----------
1. Read a dataset-level stratification label table.
2. Read predictive-performance metrics for M0-M6, including Q2, RMSECV, and MAECV.
3. Read diagnostic metrics for M0-M6, especially AE_IQR.
4. Read optimization-stability results under three-model and seven-model comparisons.
5. Summarize results by:
   - experimental design type,
   - sample-size group,
   - number-of-factors group.
6. Generate an Excel workbook containing:
   - subgroup sample counts,
   - long-format model metrics,
   - stratified median/IQR summaries,
   - model rankings within each subgroup,
   - optimization-stability summaries,
   - a compact main-text-ready summary table,
   - unmatched-file checks,
   - run logs.

Important notes
---------------
1. This script does not fit models. It only summarizes existing output files.
2. All file paths should be filled manually in the USER SETTINGS section.
3. The script tries to recognize common English column names automatically.
4. It also supports common Chinese column names through Unicode escape strings,
   but all script comments, variables, output columns, sheet names, and console
   messages are kept in English.
5. Dataset filenames are normalized before merging. For example:
   "1005.1" and "1005.1.xlsx" will be treated as the same dataset file.
6. For Q2, a higher median is ranked better. For RMSECV, MAECV, and AE_IQR,
   a lower median is ranked better.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# USER SETTINGS
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Stratification label file
# -----------------------------------------------------------------------------
# This Excel file should contain one row per dataset and at least four columns:
# dataset filename, design-type subgroup, sample-size subgroup, and factor-number subgroup.

LABEL_FILE = Path(r"PLEASE_ENTER_PATH_TO_STRATIFICATION_LABEL_FILE.xlsx")
LABEL_SHEET = 0
# Example:
# LABEL_SHEET = "Original_index_with_stratification_labels"


# -----------------------------------------------------------------------------
# 2. Output file
# -----------------------------------------------------------------------------

OUTPUT_FILE = Path(r"PLEASE_ENTER_OUTPUT_PATH\stratified_robustness_summary.xlsx")


# -----------------------------------------------------------------------------
# 3. Predictive-performance files
# -----------------------------------------------------------------------------
# Option A: use one combined long-format or wide-format file.
# The combined file may contain:
# Dataset_ID, Model, Q2, RMSECV, MAECV
# or:
# Dataset_ID, Q2_M0, Q2_M1, RMSECV_M0, RMSECV_M1, etc.

USE_COMBINED_PREDICTIVE_FILE = False

COMBINED_PREDICTIVE_FILE = Path(r"PLEASE_ENTER_PATH_TO_COMBINED_Q2_FILE.xlsx")
COMBINED_PREDICTIVE_SHEET = "All_Q2_Long"


# Option B: use separate predictive-performance files for each model.
# Each file should contain at least:
# Dataset_ID or filename + one or more of Q2, RMSECV, MAECV.

PREDICTIVE_FILES = {
    "M0": Path(r"PLEASE_ENTER_PATH_TO_M0_Q2_FILE.xlsx"),
    "M1": Path(r"PLEASE_ENTER_PATH_TO_M1_Q2_FILE.xlsx"),
    "M2": Path(r"PLEASE_ENTER_PATH_TO_M2_Q2_FILE.xlsx"),
    "M3": Path(r"PLEASE_ENTER_PATH_TO_M3_Q2_FILE.xlsx"),
    "M4": Path(r"PLEASE_ENTER_PATH_TO_M4_Q2_FILE.xlsx"),
    "M5": Path(r"PLEASE_ENTER_PATH_TO_M5_Q2_FILE.xlsx"),
    "M6": Path(r"PLEASE_ENTER_PATH_TO_M6_Q2_FILE.xlsx"),
}


# -----------------------------------------------------------------------------
# 4. Diagnostic files
# -----------------------------------------------------------------------------
# Each diagnostic file should contain at least:
# Dataset_ID or filename + AE_IQR.

DIAGNOSTIC_FILES = {
    "M0": Path(r"PLEASE_ENTER_PATH_TO_M0_DIAGNOSTIC_FILE.xlsx"),
    "M1": Path(r"PLEASE_ENTER_PATH_TO_M1_DIAGNOSTIC_FILE.xlsx"),
    "M2": Path(r"PLEASE_ENTER_PATH_TO_M2_DIAGNOSTIC_FILE.xlsx"),
    "M3": Path(r"PLEASE_ENTER_PATH_TO_M3_DIAGNOSTIC_FILE.xlsx"),
    "M4": Path(r"PLEASE_ENTER_PATH_TO_M4_DIAGNOSTIC_FILE.xlsx"),
    "M5": Path(r"PLEASE_ENTER_PATH_TO_M5_DIAGNOSTIC_FILE.xlsx"),
    "M6": Path(r"PLEASE_ENTER_PATH_TO_M6_DIAGNOSTIC_FILE.xlsx"),
}


# -----------------------------------------------------------------------------
# 5. Optimization-stability files
# -----------------------------------------------------------------------------
# Each file should contain at least:
# Dataset_ID or filename + Response_CV and/or Average_Factor_CV.
#
# M0_M1_M2: three-model comparison.
# M0_to_M6: seven-model comparison.

OPTIMIZATION_STABILITY_FILES = {
    "M0_M1_M2": Path(r"PLEASE_ENTER_PATH_TO_3MODEL_STABILITY_FILE.xlsx"),
    "M0_to_M6": Path(r"PLEASE_ENTER_PATH_TO_7MODEL_STABILITY_FILE.xlsx"),
}


# -----------------------------------------------------------------------------
# 6. General settings
# -----------------------------------------------------------------------------

MODEL_ORDER = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]

PREDICTIVE_METRICS = ["Q2", "RMSECV", "MAECV"]
DIAGNOSTIC_METRICS = ["AE_IQR"]

HIGHER_IS_BETTER = {"Q2"}
LOWER_IS_BETTER = {"RMSECV", "MAECV", "AE_IQR"}

STABILITY_METRICS = ["Response_CV", "Average_Factor_CV"]


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def is_placeholder_path(path):
    """
    Check whether a path is still a placeholder.
    """
    if path is None:
        return True

    path_string = str(path).strip()

    if path_string == "":
        return True

    placeholder_tokens = [
        "PLEASE_ENTER",
        "PLEASE_FILL",
        "YOUR_PATH",
        "PATH_TO",
    ]

    return any(token in path_string for token in placeholder_tokens)


def normalize_filename(value):
    """
    Normalize dataset filenames before merging.

    Examples
    --------
    1005.1       -> 1005.1.xlsx
    1005.1.xlsx  -> 1005.1.xlsx
    C:/data/1005.1.xlsx -> 1005.1.xlsx
    """
    if pd.isna(value):
        return np.nan

    filename = str(value).strip().replace("\\", "/")
    filename = Path(filename).name.strip()

    if filename == "":
        return np.nan

    lower_filename = filename.lower()

    if not (
        lower_filename.endswith(".xlsx")
        or lower_filename.endswith(".xls")
        or lower_filename.endswith(".csv")
    ):
        filename = filename + ".xlsx"

    return filename


def normalize_colname(name):
    """
    Normalize column names for fuzzy matching.
    """
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("–", "")
        .replace("—", "")
        .replace(".", "")
        .replace("²", "2")
        .replace("^", "")
        .replace("％", "%")
        .replace("%", "")
        .replace("（", "(")
        .replace("）", ")")
        .replace("：", ":")
    )


def find_col(df, candidates):
    """
    Find a column in a dataframe using a list of candidate names.
    """
    normalized_map = {normalize_colname(col): col for col in df.columns}

    for candidate in candidates:
        key = normalize_colname(candidate)
        if key in normalized_map:
            return normalized_map[key]

    return None


def read_excel_smart(path, sheet_name=0):
    """
    Read an Excel file using a specified sheet.

    If sheet_name is a placeholder string, the first sheet will be used.
    """
    if isinstance(sheet_name, str):
        if "PLEASE_ENTER" in sheet_name or "PLEASE_FILL" in sheet_name:
            sheet_name = 0

    return pd.read_excel(path, sheet_name=sheet_name)


def safe_numeric(series):
    """
    Convert a pandas Series to numeric values.
    """
    return pd.to_numeric(series, errors="coerce")


def proportion_lt_10(values):
    """
    Calculate the proportion of values lower than 10%.
    """
    numeric_values = safe_numeric(values).dropna()

    if len(numeric_values) == 0:
        return np.nan

    return float(np.mean(numeric_values < 0.10))


def proportion_10_to_30(values):
    """
    Calculate the proportion of values between 10% and 30%.
    """
    numeric_values = safe_numeric(values).dropna()

    if len(numeric_values) == 0:
        return np.nan

    return float(np.mean((numeric_values >= 0.10) & (numeric_values <= 0.30)))


def proportion_gt_30(values):
    """
    Calculate the proportion of values greater than 30%.
    """
    numeric_values = safe_numeric(values).dropna()

    if len(numeric_values) == 0:
        return np.nan

    return float(np.mean(numeric_values > 0.30))


def format_median_iqr(row):
    """
    Format median and interquartile range as text.
    """
    if pd.isna(row["Median"]):
        return ""

    return f"{row['Median']:.4g} ({row['Q1']:.4g}-{row['Q3']:.4g})"


# =============================================================================
# COLUMN NAME CANDIDATES
# =============================================================================

FILENAME_CANDIDATES = [
    "filename",
    "file_name",
    "file",
    "Dataset_ID",
    "dataset_id",
    "datasetid",
    "dataset",
    "dataset_name",
    "source_file",
    "sourcefile",
    "workbook",
    "\u6587\u4ef6\u540d",
    "\u6570\u636e\u96c6",
    "\u6570\u636e\u96c6\u540d\u79f0",
]

MODEL_CANDIDATES = [
    "Model",
    "model",
    "model_name",
    "\u6a21\u578b",
    "\u6a21\u578b\u540d\u79f0",
]

DESIGN_TYPE_CANDIDATES = [
    "Design_type",
    "Design type",
    "Design",
    "design_group",
    "design_stratum",
    "Experimental_design",
    "Experimental design",
    "\u8bbe\u8ba1\u7c7b\u578b\u5206\u5c42",
    "\u8bbe\u8ba1\u7c7b\u578b",
    "\u5b9e\u9a8c\u8bbe\u8ba1\u7c7b\u578b",
]

SAMPLE_SIZE_CANDIDATES = [
    "Sample_size",
    "Sample size",
    "Sample_size_group",
    "sample_group",
    "sample_stratum",
    "n_group",
    "\u6837\u672c\u91cf\u5206\u5c42",
    "\u6837\u672c\u91cf",
]

FACTOR_NUMBER_CANDIDATES = [
    "Factor_number",
    "Factor number",
    "Number_of_factors",
    "n_factors",
    "factor_group",
    "factor_stratum",
    "\u56e0\u7d20\u6570\u5206\u5c42",
    "\u56e0\u7d20\u6570",
    "\u56e0\u5b50\u6570",
]

Q2_CANDIDATES = [
    "Q2",
    "Q²",
    "q2",
    "Q_2",
]

RMSECV_CANDIDATES = [
    "RMSECV",
    "RMSE_CV",
    "RMSE CV",
    "rmsecv",
]

MAECV_CANDIDATES = [
    "MAECV",
    "MAE_CV",
    "MAE CV",
    "maecv",
]

AE_IQR_CANDIDATES = [
    "AE_IQR",
    "AE-IQR",
    "AE IQR",
    "AEIQR",
    "Abs_Error_IQR",
    "Abs Error IQR",
    "absolute_error_iqr",
    "abs_error_iqr",
    "AbsErrorIQR",
    "IQR_AE",
    "IQR of absolute errors",
    "\u7edd\u5bf9\u8bef\u5dee\u56db\u5206\u4f4d\u8ddd",
    "\u7edd\u5bf9\u8bef\u5deeIQR",
]

RESPONSE_CV_CANDIDATES = [
    "Response_CV",
    "Response CV",
    "response_cv",
    "Y_CV",
    "Y CV",
    "Optimal_Response_CV",
    "Optimal Response CV",
    "Predicted_Response_CV",
    "CV_Response",
    "CV Response",
    "\u54cd\u5e94CV",
    "\u6700\u4f18\u54cd\u5e94CV",
]

AVERAGE_FACTOR_CV_CANDIDATES = [
    "Average_Factor_CV",
    "Average Factor CV",
    "Avg_Factor_CV",
    "Avg Factor CV",
    "AvgFactorCV",
    "Factor_CV",
    "Factor CV",
    "Mean_Factor_CV",
    "Mean Factor CV",
    "Average_CV_of_Factors",
    "Average CV of Factors",
    "AverageFactorCV",
    "\u5e73\u5747\u56e0\u7d20CV",
    "\u56e0\u7d20\u5e73\u5747CV",
]


# =============================================================================
# COLUMN STANDARDIZATION
# =============================================================================

def standardize_common_columns(df):
    """
    Standardize commonly used columns in model-output files.

    Target standardized names:
    Dataset_ID, Model, Q2, RMSECV, MAECV, AE_IQR, Response_CV, Average_Factor_CV.
    """
    df = df.copy()
    rename_map = {}

    file_col = find_col(df, FILENAME_CANDIDATES)
    if file_col is not None:
        rename_map[file_col] = "Dataset_ID"

    model_col = find_col(df, MODEL_CANDIDATES)
    if model_col is not None:
        rename_map[model_col] = "Model"

    q2_col = find_col(df, Q2_CANDIDATES)
    if q2_col is not None:
        rename_map[q2_col] = "Q2"

    rmsecv_col = find_col(df, RMSECV_CANDIDATES)
    if rmsecv_col is not None:
        rename_map[rmsecv_col] = "RMSECV"

    maecv_col = find_col(df, MAECV_CANDIDATES)
    if maecv_col is not None:
        rename_map[maecv_col] = "MAECV"

    ae_iqr_col = find_col(df, AE_IQR_CANDIDATES)
    if ae_iqr_col is not None:
        rename_map[ae_iqr_col] = "AE_IQR"

    response_cv_col = find_col(df, RESPONSE_CV_CANDIDATES)
    if response_cv_col is not None:
        rename_map[response_cv_col] = "Response_CV"

    average_factor_cv_col = find_col(df, AVERAGE_FACTOR_CV_CANDIDATES)
    if average_factor_cv_col is not None:
        rename_map[average_factor_cv_col] = "Average_Factor_CV"

    df = df.rename(columns=rename_map)

    if "Dataset_ID" in df.columns:
        df["Dataset_ID"] = df["Dataset_ID"].apply(normalize_filename)

    if "Model" in df.columns:
        df["Model"] = df["Model"].astype(str).str.strip()

    return df


def standardize_label_columns(df):
    """
    Standardize columns in the stratification-label file.

    Target standardized names:
    Dataset_ID, Design_type, Sample_size, Factor_number.
    """
    df = df.copy()
    rename_map = {}

    file_col = find_col(df, FILENAME_CANDIDATES)
    if file_col is not None:
        rename_map[file_col] = "Dataset_ID"

    design_col = find_col(df, DESIGN_TYPE_CANDIDATES)
    if design_col is not None:
        rename_map[design_col] = "Design_type"

    sample_col = find_col(df, SAMPLE_SIZE_CANDIDATES)
    if sample_col is not None:
        rename_map[sample_col] = "Sample_size"

    factor_col = find_col(df, FACTOR_NUMBER_CANDIDATES)
    if factor_col is not None:
        rename_map[factor_col] = "Factor_number"

    df = df.rename(columns=rename_map)

    if "Dataset_ID" in df.columns:
        df["Dataset_ID"] = df["Dataset_ID"].apply(normalize_filename)

    return df


# =============================================================================
# READING MODEL METRICS
# =============================================================================

def read_separate_model_metrics(path, model_name, metric_cols):
    """
    Read one model-specific metric file and return long-format data.

    Output columns:
    Dataset_ID, Model, Metric, Value.
    """
    logs = []

    if is_placeholder_path(path):
        logs.append(f"{model_name}: path is not configured, skipped.")
        return pd.DataFrame(), logs

    path = Path(path)

    if not path.exists():
        logs.append(f"{model_name}: file not found, skipped: {path}")
        return pd.DataFrame(), logs

    df_raw = read_excel_smart(path, sheet_name=0)
    raw_cols = list(df_raw.columns)

    df = standardize_common_columns(df_raw)

    if "Dataset_ID" not in df.columns:
        logs.append(
            f"{model_name}: Dataset_ID column not found, skipped. "
            f"File: {path}; raw columns: {raw_cols}; standardized columns: {list(df.columns)}"
        )
        return pd.DataFrame(), logs

    available_metrics = [metric for metric in metric_cols if metric in df.columns]

    if not available_metrics:
        logs.append(
            f"{model_name}: target metric columns not found, skipped. "
            f"Expected: {metric_cols}; file: {path}; raw columns: {raw_cols}; "
            f"standardized columns: {list(df.columns)}"
        )
        return pd.DataFrame(), logs

    df = df[["Dataset_ID"] + available_metrics].copy()
    df["Model"] = model_name

    long_df = df.melt(
        id_vars=["Dataset_ID", "Model"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Value",
    )

    long_df["Value"] = safe_numeric(long_df["Value"])

    logs.append(
        f"{model_name}: successfully read {path.name}; "
        f"metrics={available_metrics}; rows={df.shape[0]}"
    )

    return long_df, logs


def read_combined_model_metrics(path, sheet_name, metric_cols):
    """
    Read a combined model metric file.

    Supported formats
    -----------------
    1. Long format:
       Dataset_ID, Model, Q2, RMSECV, MAECV.

    2. Wide format:
       Dataset_ID, Q2_M0, Q2_M1, RMSECV_M0, RMSECV_M1, etc.
    """
    logs = []

    if is_placeholder_path(path):
        logs.append("Combined predictive file path is not configured, skipped.")
        return pd.DataFrame(), logs

    path = Path(path)

    if not path.exists():
        logs.append(f"Combined predictive file not found, skipped: {path}")
        return pd.DataFrame(), logs

    df_raw = read_excel_smart(path, sheet_name=sheet_name)
    raw_cols = list(df_raw.columns)

    df = standardize_common_columns(df_raw)

    if "Dataset_ID" not in df.columns:
        logs.append(
            f"Combined predictive file: Dataset_ID column not found, skipped. "
            f"File: {path}; raw columns: {raw_cols}; standardized columns: {list(df.columns)}"
        )
        return pd.DataFrame(), logs

    if "Model" in df.columns:
        available_metrics = [metric for metric in metric_cols if metric in df.columns]

        if available_metrics:
            keep_cols = ["Dataset_ID", "Model"] + available_metrics
            out = df[keep_cols].copy()

            long_df = out.melt(
                id_vars=["Dataset_ID", "Model"],
                value_vars=available_metrics,
                var_name="Metric",
                value_name="Value",
            )

            long_df["Value"] = safe_numeric(long_df["Value"])

            logs.append(
                f"Combined predictive file read as long format: {path.name}; "
                f"metrics={available_metrics}; rows={out.shape[0]}"
            )

            return long_df, logs

    rows = []
    normalized_columns = {col: normalize_colname(col) for col in df.columns}

    for col in df.columns:
        if col == "Dataset_ID":
            continue

        col_key = normalized_columns[col]

        matched_metric = None
        matched_model = None

        for metric in metric_cols:
            for model in MODEL_ORDER:
                possible_names = [
                    f"{metric}_{model}",
                    f"{metric}{model}",
                    f"{model}_{metric}",
                    f"{model}{metric}",
                ]

                possible_keys = [normalize_colname(name) for name in possible_names]

                if col_key in possible_keys:
                    matched_metric = metric
                    matched_model = model
                    break

            if matched_metric is not None:
                break

        if matched_metric is not None and matched_model is not None:
            tmp = df[["Dataset_ID", col]].copy()
            tmp["Model"] = matched_model
            tmp["Metric"] = matched_metric
            tmp["Value"] = safe_numeric(tmp[col])
            tmp = tmp[["Dataset_ID", "Model", "Metric", "Value"]]
            rows.append(tmp)

    if rows:
        long_df = pd.concat(rows, ignore_index=True)

        logs.append(
            f"Combined predictive file read as wide format: {path.name}; "
            f"rows={long_df.shape[0]}"
        )

        return long_df, logs

    logs.append(
        f"Combined predictive file could not be parsed. "
        f"File: {path}; raw columns: {raw_cols}; standardized columns: {list(df.columns)}"
    )

    return pd.DataFrame(), logs


def read_predictive_metrics():
    """
    Read predictive metrics according to the selected input mode.
    """
    logs = []

    if USE_COMBINED_PREDICTIVE_FILE:
        predictive_long, msg = read_combined_model_metrics(
            path=COMBINED_PREDICTIVE_FILE,
            sheet_name=COMBINED_PREDICTIVE_SHEET,
            metric_cols=PREDICTIVE_METRICS,
        )
        logs.extend(msg)

    else:
        parts = []

        for model_name, path in PREDICTIVE_FILES.items():
            df_long, msg = read_separate_model_metrics(
                path=path,
                model_name=model_name,
                metric_cols=PREDICTIVE_METRICS,
            )
            logs.extend(msg)

            if not df_long.empty:
                parts.append(df_long)

        if parts:
            predictive_long = pd.concat(parts, ignore_index=True)
        else:
            predictive_long = pd.DataFrame(
                columns=["Dataset_ID", "Model", "Metric", "Value"]
            )

    logs.append(f"Predictive metrics merged: rows={predictive_long.shape[0]}")

    return predictive_long, logs


def read_diagnostic_metrics():
    """
    Read diagnostic metrics from separate model-specific files.
    """
    logs = []
    parts = []

    for model_name, path in DIAGNOSTIC_FILES.items():
        df_long, msg = read_separate_model_metrics(
            path=path,
            model_name=model_name,
            metric_cols=DIAGNOSTIC_METRICS,
        )
        logs.extend(msg)

        if not df_long.empty:
            parts.append(df_long)

    if parts:
        diagnostic_long = pd.concat(parts, ignore_index=True)
    else:
        diagnostic_long = pd.DataFrame(
            columns=["Dataset_ID", "Model", "Metric", "Value"]
        )

    logs.append(f"Diagnostic metrics merged: rows={diagnostic_long.shape[0]}")

    return diagnostic_long, logs


# =============================================================================
# READING STRATIFICATION LABELS
# =============================================================================

def read_stratification_labels():
    """
    Read the stratification label file and convert it into both wide and long formats.

    labels_wide columns:
    Dataset_ID, Design_type, Sample_size, Factor_number.

    label_long columns:
    Dataset_ID, Stratification, Subgroup.
    """
    logs = []

    if is_placeholder_path(LABEL_FILE):
        raise ValueError("Please fill LABEL_FILE in the USER SETTINGS section.")

    label_path = Path(LABEL_FILE)

    if not label_path.exists():
        raise FileNotFoundError(f"Stratification label file not found: {label_path}")

    labels_raw = read_excel_smart(label_path, sheet_name=LABEL_SHEET)
    labels = standardize_label_columns(labels_raw)

    required_cols = ["Dataset_ID", "Design_type", "Sample_size", "Factor_number"]
    missing_cols = [col for col in required_cols if col not in labels.columns]

    if missing_cols:
        raise ValueError(
            f"Stratification label file is missing required columns: {missing_cols}\n"
            f"Current columns: {list(labels.columns)}"
        )

    labels = labels[required_cols].copy()

    logs.append(
        f"Stratification labels successfully read: {label_path.name}; "
        f"rows={labels.shape[0]}"
    )

    label_long_parts = []

    strat_map = {
        "Design_type": "Design_type",
        "Sample_size": "Sample_size",
        "Factor_number": "Factor_number",
    }

    for stratification_name, col in strat_map.items():
        tmp = labels[["Dataset_ID", col]].copy()
        tmp = tmp.rename(columns={col: "Subgroup"})
        tmp["Stratification"] = stratification_name
        tmp = tmp[["Dataset_ID", "Stratification", "Subgroup"]]
        label_long_parts.append(tmp)

    label_long = pd.concat(label_long_parts, ignore_index=True)

    subgroup_counts = (
        label_long
        .groupby(["Stratification", "Subgroup"], dropna=False)
        .agg(N_datasets=("Dataset_ID", "nunique"))
        .reset_index()
        .sort_values(["Stratification", "Subgroup"])
    )

    return labels, label_long, subgroup_counts, logs


# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================

def summarize_model_metrics(df):
    """
    Summarize model metrics by stratification, subgroup, model, and metric.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["Value"] = safe_numeric(df["Value"])

    summary = (
        df.groupby(
            ["Stratification", "Subgroup", "Model", "Metric"],
            dropna=False
        )
        .agg(
            N_datasets=("Dataset_ID", "nunique"),
            N_non_missing=("Value", lambda x: safe_numeric(x).notna().sum()),
            Median=("Value", "median"),
            Q1=("Value", lambda x: safe_numeric(x).quantile(0.25)),
            Q3=("Value", lambda x: safe_numeric(x).quantile(0.75)),
            Mean=("Value", "mean"),
            SD=("Value", "std"),
            Min=("Value", "min"),
            Max=("Value", "max"),
        )
        .reset_index()
    )

    summary["IQR"] = summary["Q3"] - summary["Q1"]
    summary["Median_IQR"] = summary.apply(format_median_iqr, axis=1)

    ranked_parts = []

    for (strat, subgroup, metric), sub in summary.groupby(
        ["Stratification", "Subgroup", "Metric"],
        dropna=False
    ):
        sub = sub.copy()

        if metric in HIGHER_IS_BETTER:
            sub["Rank"] = sub["Median"].rank(ascending=False, method="min")
        elif metric in LOWER_IS_BETTER:
            sub["Rank"] = sub["Median"].rank(ascending=True, method="min")
        else:
            sub["Rank"] = np.nan

        ranked_parts.append(sub)

    if ranked_parts:
        summary = pd.concat(ranked_parts, ignore_index=True)

    summary = summary.sort_values(
        ["Stratification", "Subgroup", "Metric", "Rank", "Model"],
        na_position="last"
    )

    return summary


def summarize_stability(df):
    """
    Summarize optimization-stability metrics by stratification and subgroup.
    """
    if df.empty:
        return pd.DataFrame()

    value_cols = [col for col in STABILITY_METRICS if col in df.columns]

    if not value_cols:
        return pd.DataFrame()

    rows = []

    for col in value_cols:
        tmp = (
            df.groupby(
                ["Stratification", "Subgroup", "Comparison"],
                dropna=False
            )
            .agg(
                N_datasets=("Dataset_ID", "nunique"),
                N_non_missing=(col, lambda x: safe_numeric(x).notna().sum()),
                Median=(col, "median"),
                Q1=(col, lambda x: safe_numeric(x).quantile(0.25)),
                Q3=(col, lambda x: safe_numeric(x).quantile(0.75)),
                Mean=(col, "mean"),
                SD=(col, "std"),
                Min=(col, "min"),
                Max=(col, "max"),
                Proportion_lt_10pct=(col, proportion_lt_10),
                Proportion_10_30pct=(col, proportion_10_to_30),
                Proportion_gt_30pct=(col, proportion_gt_30),
            )
            .reset_index()
        )

        tmp["Metric"] = col
        tmp["IQR"] = tmp["Q3"] - tmp["Q1"]
        tmp["Median_IQR"] = tmp.apply(format_median_iqr, axis=1)

        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)

    out = out.sort_values(
        ["Stratification", "Subgroup", "Metric", "Comparison"]
    )

    return out


def make_rank_wide(model_summary):
    """
    Convert model-ranking results into wide format.
    """
    if model_summary.empty:
        return pd.DataFrame()

    rank_wide = (
        model_summary
        .pivot_table(
            index=["Stratification", "Subgroup", "Metric"],
            columns="Model",
            values="Rank",
            aggfunc="first"
        )
        .reset_index()
    )

    return rank_wide


def make_median_wide(model_summary):
    """
    Convert median-IQR results into wide format.
    """
    if model_summary.empty:
        return pd.DataFrame()

    median_wide = (
        model_summary
        .pivot_table(
            index=["Stratification", "Subgroup", "Metric"],
            columns="Model",
            values="Median_IQR",
            aggfunc="first"
        )
        .reset_index()
    )

    return median_wide


def make_core_interpretation(model_summary, stability_summary):
    """
    Generate a compact interpretation table for checking the main stratified patterns.
    """
    rows = []

    if not model_summary.empty:
        for metric in ["Q2", "RMSECV", "MAECV", "AE_IQR"]:
            sub_metric = model_summary[model_summary["Metric"] == metric].copy()

            if sub_metric.empty:
                continue

            best_rows = (
                sub_metric
                .sort_values(["Stratification", "Subgroup", "Rank"])
                .groupby(["Stratification", "Subgroup"], as_index=False)
                .first()
            )

            for _, row in best_rows.iterrows():
                rows.append({
                    "Stratification": row["Stratification"],
                    "Subgroup": row["Subgroup"],
                    "Check_item": f"Best model by median {metric}",
                    "Result": row["Model"],
                    "Value": row["Median"],
                    "Interpretation": (
                        "This item identifies the best-ranked model within each subgroup "
                        "based on the median value of the selected metric."
                    ),
                })

    if not stability_summary.empty:
        for metric in ["Response_CV", "Average_Factor_CV"]:
            sub_metric = stability_summary[
                stability_summary["Metric"] == metric
            ].copy()

            if sub_metric.empty:
                continue

            for (strat, subgroup), sub in sub_metric.groupby(
                ["Stratification", "Subgroup"],
                dropna=False
            ):
                comp_map = dict(zip(sub["Comparison"], sub["Median"]))

                if "M0_M1_M2" in comp_map and "M0_to_M6" in comp_map:
                    value_three = comp_map["M0_M1_M2"]
                    value_seven = comp_map["M0_to_M6"]

                    ratio = np.nan
                    if pd.notna(value_three) and value_three != 0:
                        ratio = value_seven / value_three

                    rows.append({
                        "Stratification": strat,
                        "Subgroup": subgroup,
                        "Check_item": f"{metric}: seven-model vs three-model",
                        "Result": (
                            f"M0_to_M6={value_seven:.4g}; "
                            f"M0_M1_M2={value_three:.4g}; "
                            f"ratio={ratio:.3g}"
                        ),
                        "Value": ratio,
                        "Interpretation": (
                            "A higher seven-model value suggests that expanding the "
                            "comparison from traditional regression models to all candidate "
                            "models increases optimization-conclusion dispersion."
                        ),
                    })

    return pd.DataFrame(rows)


def make_main_summary_for_text(subgroup_counts, model_summary, stability_summary):
    """
    Generate a compact table suitable for main-text description.
    """
    rows = []

    for _, count_row in subgroup_counts.iterrows():
        strat = count_row["Stratification"]
        subgroup = count_row["Subgroup"]
        n_datasets = count_row["N_datasets"]

        row = {
            "Stratification": strat,
            "Subgroup": subgroup,
            "N_datasets": n_datasets,
            "Best_Q2_model": "",
            "Best_Q2_median": np.nan,
            "Best_RMSECV_model": "",
            "Best_RMSECV_median": np.nan,
            "Best_MAECV_model": "",
            "Best_MAECV_median": np.nan,
            "Best_AE_IQR_model": "",
            "Best_AE_IQR_median": np.nan,
            "Response_CV_3models_median": np.nan,
            "Response_CV_7models_median": np.nan,
            "Average_Factor_CV_3models_median": np.nan,
            "Average_Factor_CV_7models_median": np.nan,
            "Overall_note": "",
        }

        for metric in ["Q2", "RMSECV", "MAECV", "AE_IQR"]:
            if model_summary.empty:
                continue

            sub = model_summary[
                (model_summary["Stratification"] == strat)
                & (model_summary["Subgroup"] == subgroup)
                & (model_summary["Metric"] == metric)
            ].copy()

            if sub.empty:
                continue

            sub = sub.sort_values("Rank")
            best = sub.iloc[0]

            row[f"Best_{metric}_model"] = best["Model"]
            row[f"Best_{metric}_median"] = best["Median"]

        if not stability_summary.empty:
            for metric in ["Response_CV", "Average_Factor_CV"]:
                sub = stability_summary[
                    (stability_summary["Stratification"] == strat)
                    & (stability_summary["Subgroup"] == subgroup)
                    & (stability_summary["Metric"] == metric)
                ].copy()

                if sub.empty:
                    continue

                comp_map = dict(zip(sub["Comparison"], sub["Median"]))

                if metric == "Response_CV":
                    row["Response_CV_3models_median"] = comp_map.get(
                        "M0_M1_M2",
                        np.nan
                    )
                    row["Response_CV_7models_median"] = comp_map.get(
                        "M0_to_M6",
                        np.nan
                    )

                elif metric == "Average_Factor_CV":
                    row["Average_Factor_CV_3models_median"] = comp_map.get(
                        "M0_M1_M2",
                        np.nan
                    )
                    row["Average_Factor_CV_7models_median"] = comp_map.get(
                        "M0_to_M6",
                        np.nan
                    )

        notes = []

        if row["Best_Q2_model"] in ["M1", "M2", "M4"]:
            notes.append("predictive pattern broadly consistent")
        elif row["Best_Q2_model"] != "":
            notes.append("predictive ranking differs")

        if (
            pd.notna(row["Response_CV_3models_median"])
            and pd.notna(row["Response_CV_7models_median"])
        ):
            if row["Response_CV_7models_median"] > row["Response_CV_3models_median"]:
                notes.append("seven-model response dispersion higher")
            else:
                notes.append("seven-model response dispersion not higher")

        if (
            pd.notna(row["Average_Factor_CV_3models_median"])
            and pd.notna(row["Average_Factor_CV_7models_median"])
        ):
            if (
                row["Average_Factor_CV_7models_median"]
                > row["Average_Factor_CV_3models_median"]
            ):
                notes.append("seven-model factor dispersion higher")
            else:
                notes.append("seven-model factor dispersion not higher")

        row["Overall_note"] = "; ".join(notes)

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# READING OPTIMIZATION-STABILITY FILES
# =============================================================================

def read_stability_files():
    """
    Read optimization-stability files and return a combined dataframe.
    """
    logs = []
    parts = []

    for comparison_name, path in OPTIMIZATION_STABILITY_FILES.items():
        if is_placeholder_path(path):
            logs.append(f"{comparison_name}: path is not configured, skipped.")
            continue

        path = Path(path)

        if not path.exists():
            logs.append(f"{comparison_name}: file not found, skipped: {path}")
            continue

        df_raw = read_excel_smart(path, sheet_name=0)
        raw_cols = list(df_raw.columns)

        df = standardize_common_columns(df_raw)

        if "Dataset_ID" not in df.columns:
            logs.append(
                f"{comparison_name}: Dataset_ID column not found, skipped. "
                f"File: {path}; raw columns: {raw_cols}; "
                f"standardized columns: {list(df.columns)}"
            )
            continue

        value_cols = [col for col in STABILITY_METRICS if col in df.columns]

        if not value_cols:
            logs.append(
                f"{comparison_name}: stability metric columns not found, skipped. "
                f"Expected: {STABILITY_METRICS}; file: {path}; raw columns: {raw_cols}; "
                f"standardized columns: {list(df.columns)}"
            )
            continue

        keep_cols = ["Dataset_ID"] + value_cols
        df = df[keep_cols].copy()

        for col in value_cols:
            df[col] = safe_numeric(df[col])

        df["Comparison"] = comparison_name

        parts.append(df)

        logs.append(
            f"{comparison_name}: successfully read {path.name}; "
            f"metrics={value_cols}; rows={df.shape[0]}"
        )

    if parts:
        stability_all = pd.concat(parts, ignore_index=True)
    else:
        stability_all = pd.DataFrame(
            columns=["Dataset_ID", "Response_CV", "Average_Factor_CV", "Comparison"]
        )

    logs.append(f"Optimization-stability files merged: rows={stability_all.shape[0]}")

    return stability_all, logs


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """
    Run the stratified robustness summary workflow.
    """
    logs = []

    labels, label_long, subgroup_counts, msg = read_stratification_labels()
    logs.extend(msg)

    predictive_long, msg = read_predictive_metrics()
    logs.extend(msg)

    diagnostic_long, msg = read_diagnostic_metrics()
    logs.extend(msg)

    model_metrics_long = pd.concat(
        [predictive_long, diagnostic_long],
        ignore_index=True
    )

    logs.append(f"All model metrics merged: rows={model_metrics_long.shape[0]}")

    if not model_metrics_long.empty:
        model_metrics_with_strata = model_metrics_long.merge(
            label_long,
            on="Dataset_ID",
            how="left",
        )

        unmatched_model_metrics = model_metrics_with_strata[
            model_metrics_with_strata["Stratification"].isna()
        ].copy()

        model_summary = summarize_model_metrics(model_metrics_with_strata)
        model_median_wide = make_median_wide(model_summary)
        model_rank_wide = make_rank_wide(model_summary)

    else:
        model_metrics_with_strata = pd.DataFrame()
        unmatched_model_metrics = pd.DataFrame()
        model_summary = pd.DataFrame()
        model_median_wide = pd.DataFrame()
        model_rank_wide = pd.DataFrame()

    logs.append(f"Model-metric stratified summary completed: rows={model_summary.shape[0]}")

    stability_all, msg = read_stability_files()
    logs.extend(msg)

    if not stability_all.empty:
        stability_with_strata = stability_all.merge(
            label_long,
            on="Dataset_ID",
            how="left",
        )

        unmatched_stability = stability_with_strata[
            stability_with_strata["Stratification"].isna()
        ].copy()

        stability_summary = summarize_stability(stability_with_strata)

    else:
        stability_with_strata = pd.DataFrame()
        unmatched_stability = pd.DataFrame()
        stability_summary = pd.DataFrame()

    logs.append(
        f"Optimization-stability stratified summary completed: "
        f"rows={stability_summary.shape[0]}"
    )

    auto_interpretation = make_core_interpretation(
        model_summary=model_summary,
        stability_summary=stability_summary,
    )

    main_summary = make_main_summary_for_text(
        subgroup_counts=subgroup_counts,
        model_summary=model_summary,
        stability_summary=stability_summary,
    )

    if is_placeholder_path(OUTPUT_FILE):
        raise ValueError("Please fill OUTPUT_FILE in the USER SETTINGS section.")

    output_file = Path(OUTPUT_FILE)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log_df = pd.DataFrame({"Log": logs})

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        subgroup_counts.to_excel(
            writer,
            sheet_name="01_subgroup_counts",
            index=False,
        )

        labels.to_excel(
            writer,
            sheet_name="02_labels_used",
            index=False,
        )

        if not model_metrics_with_strata.empty:
            model_metrics_with_strata.to_excel(
                writer,
                sheet_name="03_model_metrics_long",
                index=False,
            )

        if not model_summary.empty:
            model_summary.to_excel(
                writer,
                sheet_name="04_model_summary",
                index=False,
            )

        if not model_median_wide.empty:
            model_median_wide.to_excel(
                writer,
                sheet_name="05_model_median_wide",
                index=False,
            )

        if not model_rank_wide.empty:
            model_rank_wide.to_excel(
                writer,
                sheet_name="06_model_rank_wide",
                index=False,
            )

        if not stability_with_strata.empty:
            stability_with_strata.to_excel(
                writer,
                sheet_name="07_stability_long",
                index=False,
            )

        if not stability_summary.empty:
            stability_summary.to_excel(
                writer,
                sheet_name="08_stability_summary",
                index=False,
            )

        if not main_summary.empty:
            main_summary.to_excel(
                writer,
                sheet_name="09_main_summary_text",
                index=False,
            )

        if not auto_interpretation.empty:
            auto_interpretation.to_excel(
                writer,
                sheet_name="10_auto_interpretation",
                index=False,
            )

        if not unmatched_model_metrics.empty:
            unmatched_model_metrics.to_excel(
                writer,
                sheet_name="11_unmatched_model",
                index=False,
            )

        if not unmatched_stability.empty:
            unmatched_stability.to_excel(
                writer,
                sheet_name="12_unmatched_stability",
                index=False,
            )

        log_df.to_excel(
            writer,
            sheet_name="13_run_log",
            index=False,
        )

    print("=" * 90)
    print("Stratified robustness summary completed")
    print("=" * 90)
    print(f"Output file: {output_file}")
    print("-" * 90)

    print("\nSubgroup sample counts:")
    print(subgroup_counts.to_string(index=False))

    print("\nRun log:")
    for item in logs:
        print(" -", item)

    print("-" * 90)

    if not unmatched_model_metrics.empty:
        n_unmatched = unmatched_model_metrics["Dataset_ID"].nunique()
        print(
            f"Warning: {n_unmatched} model-metric dataset filenames "
            f"were not matched to stratification labels."
        )
        print("Please check sheet: 11_unmatched_model")

    if not unmatched_stability.empty:
        n_unmatched = unmatched_stability["Dataset_ID"].nunique()
        print(
            f"Warning: {n_unmatched} stability dataset filenames "
            f"were not matched to stratification labels."
        )
        print("Please check sheet: 12_unmatched_stability")

    print("\nMost important output sheets:")
    print("01_subgroup_counts      : dataset counts by subgroup")
    print("04_model_summary        : median/IQR/rank for each model and metric")
    print("05_model_median_wide    : appendix-ready median(IQR) table")
    print("06_model_rank_wide      : model ranking table")
    print("08_stability_summary    : Response_CV and Average_Factor_CV summaries")
    print("09_main_summary_text    : compact table for main-text writing")
    print("13_run_log              : file-reading and column-matching log")
    print("=" * 90)


if __name__ == "__main__":
    main()