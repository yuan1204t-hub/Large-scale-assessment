# -*- coding: utf-8 -*-
"""
Script name
-----------
run_ml_sensitivity_analysis.py

Purpose
-------
This script performs a machine-learning-only sensitivity analysis for datasets
that were structurally excluded from the complete M0-M6 regression comparison.

Main tasks
----------
1. Read extraction-process datasets from a user-specified folder.
2. Treat the first n-1 columns as predictors and the last column as the response.
3. Evaluate four machine learning models:
   - M3: quadratic Ridge regression
   - M4: support vector regression
   - M5: partial least squares regression
   - M6: Gaussian process regression
4. Use leave-one-out cross-validation to calculate:
   - Q2
   - RMSECV
   - MAECV
5. For each dataset, identify the best-performing model according to:
   - highest Q2
   - lowest RMSECV
   - lowest MAECV
6. Summarize model-level performance and best-model counts.
7. Export all results to one Excel workbook.

Important notes
---------------
1. This script is designed for the machine-learning-only sensitivity analysis
   of structurally excluded datasets.
2. It does not run M0-M2 because these datasets do not satisfy the structural
   requirement for unified full quadratic regression comparison.
3. All file paths should be filled manually in the USER SETTINGS section.
4. By default, hyperparameter tuning is performed within each outer LOOCV
   training fold to reduce information leakage.
5. The script may be computationally intensive, especially for Gaussian process
   regression. If necessary, reduce the GPR kernel grid or set N_JOBS to a
   smaller value.
"""

import ast
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# USER SETTINGS
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Input folder
# -----------------------------------------------------------------------------
# This folder should contain the structurally excluded datasets.
# Each dataset should be an Excel or CSV file.
# For each dataset, the first n-1 columns are treated as predictors and the final
# column is treated as the response variable.

DATA_DIR = Path(r"PLEASE_ENTER_PATH_TO_STRUCTURALLY_EXCLUDED_DATASETS")


# -----------------------------------------------------------------------------
# 2. Output file
# -----------------------------------------------------------------------------

OUTPUT_FILE = Path(r"PLEASE_ENTER_OUTPUT_PATH\ml_sensitivity_analysis_results.xlsx")


# -----------------------------------------------------------------------------
# 3. Analysis settings
# -----------------------------------------------------------------------------

ANALYSIS_GROUP_NAME = "structurally_excluded_machine_learning_sensitivity"

MODEL_ORDER = ["M3_Ridge", "M4_SVR", "M5_PLS", "M6_GPR"]

PREDICTIVE_METRICS = ["Q2", "RMSECV", "MAECV"]

# If True, grid search is repeated inside each outer LOOCV training fold.
# This is more rigorous but slower.
# If False, the script first tunes hyperparameters on the full dataset and then
# uses the selected model setting for LOOCV prediction.
USE_NESTED_LOOCV_TUNING = True

# Number of parallel jobs for GridSearchCV.
# Use -1 to use all available CPU cores.
N_JOBS = -1

# Save a temporary checkpoint every N datasets.
CHECKPOINT_INTERVAL = 10

# Whether to save temporary checkpoint files during the run.
SAVE_CHECKPOINTS = True

# Whether to skip Excel files whose names look like previous result files.
SKIP_RESULT_LIKE_FILES = True


# -----------------------------------------------------------------------------
# 4. Hyperparameter grids
# -----------------------------------------------------------------------------

RIDGE_ALPHA_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
RIDGE_FIT_INTERCEPT_GRID = [True, False]

SVR_C_GRID = [0.1, 1, 10, 50, 100, 500]
SVR_EPSILON_GRID = [0.001, 0.01, 0.05, 0.1, 0.2]
SVR_GAMMA_GRID = ["scale", "auto", 0.001, 0.01, 0.1]

GPR_ALPHA_GRID = [1e-10, 1e-6, 1e-4, 1e-2]
GPR_NORMALIZE_Y_GRID = [True, False]

GPR_N_RESTARTS_OPTIMIZER = 2
GPR_RANDOM_STATE = 42


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
    Normalize dataset filenames for reporting.
    """
    if pd.isna(value):
        return np.nan

    filename = str(value).strip().replace("\\", "/")
    filename = Path(filename).name.strip()

    return filename if filename else np.nan


def get_data_files(folder):
    """
    Get valid dataset files from the input folder.
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Input data folder not found: {folder}")

    valid_suffixes = {".xlsx", ".xls", ".csv"}
    result_like_keywords = [
        "success",
        "failed",
        "log",
        "summary",
        "result",
        "results",
        "crosstab",
        "checkpoint",
        "temp",
    ]

    files = []

    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue

        if file_path.name.startswith("~$"):
            continue

        if file_path.suffix.lower() not in valid_suffixes:
            continue

        if SKIP_RESULT_LIKE_FILES:
            lower_name = file_path.name.lower()
            if any(keyword in lower_name for keyword in result_like_keywords):
                continue

        files.append(file_path)

    return sorted(files)


def read_dataset(file_path):
    """
    Read one dataset.

    By default:
    - all columns except the last one are predictors;
    - the last column is the response variable.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path, sheet_name=0)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 2:
        raise ValueError("At least one predictor column and one response column are required.")

    X_df = df.iloc[:, :-1].copy()
    y_series = df.iloc[:, -1].copy()

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    y_series = pd.to_numeric(y_series, errors="coerce")

    valid_mask = ~(X_df.isna().any(axis=1) | y_series.isna())

    X_df = X_df.loc[valid_mask].reset_index(drop=True)
    y_series = y_series.loc[valid_mask].reset_index(drop=True)

    X = X_df.values.astype(float)
    y = y_series.values.astype(float)

    return X, y


def check_ml_eligibility(X, y):
    """
    Check whether a dataset is eligible for machine-learning LOOCV.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if X.ndim != 2:
        return False, "invalid_X_dimension"

    n_samples, n_features = X.shape

    if n_samples < 3:
        return False, "too_few_samples"

    if n_features < 1:
        return False, "no_predictor"

    if np.any(pd.isna(X)) or np.any(pd.isna(y)):
        return False, "missing_values"

    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
        return False, "non_finite_values"

    if np.nanstd(y) == 0:
        return False, "constant_response"

    return True, "ok"


def calculate_q2_rmsecv_maecv(y_true, y_pred):
    """
    Calculate Q2, RMSECV, and MAECV.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if valid_mask.sum() == 0:
        return np.nan, np.nan, np.nan

    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if np.isclose(ss_tot, 0):
        q2 = np.nan
    else:
        q2 = 1 - ss_res / ss_tot

    rmsecv = np.sqrt(mean_squared_error(y_true, y_pred))
    maecv = mean_absolute_error(y_true, y_pred)

    return q2, rmsecv, maecv


def params_to_string(params):
    """
    Convert best-parameter dictionaries to a safe string.
    """
    if params is None:
        return ""

    try:
        return json.dumps(params, default=str, ensure_ascii=True)
    except Exception:
        return str(params)


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def make_gpr_kernels():
    """
    Create the candidate kernel list for Gaussian process regression.
    """
    base_kernels = [
        ConstantKernel(1.0) * RBF(length_scale=1.0),
        ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5),
        ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
        ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0),
    ]

    white_kernels = [
        ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3),
        ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-3),
        ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-3),
        ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=1e-3),
    ]

    return base_kernels + white_kernels


def get_models_and_grids(n_samples, n_features):
    """
    Define M3-M6 models and their hyperparameter grids.

    Parameters
    ----------
    n_samples:
        Number of samples in the full dataset.

    n_features:
        Number of original predictor variables.
    """
    models = {}

    # -------------------------------------------------------------------------
    # M3: quadratic Ridge regression
    # -------------------------------------------------------------------------
    ridge_pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ])

    ridge_grid = {
        "model__alpha": RIDGE_ALPHA_GRID,
        "model__fit_intercept": RIDGE_FIT_INTERCEPT_GRID,
    }

    models["M3_Ridge"] = {
        "estimator": ridge_pipeline,
        "param_grid": ridge_grid,
    }

    # -------------------------------------------------------------------------
    # M4: support vector regression
    # -------------------------------------------------------------------------
    svr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR()),
    ])

    svr_grid = [
        {
            "model__kernel": ["linear"],
            "model__C": SVR_C_GRID,
            "model__epsilon": SVR_EPSILON_GRID,
        },
        {
            "model__kernel": ["rbf"],
            "model__C": SVR_C_GRID,
            "model__epsilon": SVR_EPSILON_GRID,
            "model__gamma": SVR_GAMMA_GRID,
        },
    ]

    models["M4_SVR"] = {
        "estimator": svr_pipeline,
        "param_grid": svr_grid,
    }

    # -------------------------------------------------------------------------
    # M5: partial least squares regression
    # -------------------------------------------------------------------------
    # If nested LOOCV is used, the smallest inner training set has n_samples - 2
    # samples. The upper limit for n_components is therefore restricted to avoid
    # invalid folds.
    if USE_NESTED_LOOCV_TUNING:
        max_components = max(1, min(n_features, n_samples - 2))
    else:
        max_components = max(1, min(n_features, n_samples - 1))

    pls_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", PLSRegression(scale=False)),
    ])

    pls_grid = {
        "model__n_components": list(range(1, max_components + 1)),
    }

    models["M5_PLS"] = {
        "estimator": pls_pipeline,
        "param_grid": pls_grid,
    }

    # -------------------------------------------------------------------------
    # M6: Gaussian process regression
    # -------------------------------------------------------------------------
    gpr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianProcessRegressor(
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=GPR_N_RESTARTS_OPTIMIZER,
            random_state=GPR_RANDOM_STATE,
        )),
    ])

    gpr_grid = {
        "model__kernel": make_gpr_kernels(),
        "model__alpha": GPR_ALPHA_GRID,
        "model__normalize_y": GPR_NORMALIZE_Y_GRID,
    }

    models["M6_GPR"] = {
        "estimator": gpr_pipeline,
        "param_grid": gpr_grid,
    }

    return models


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def tune_model_on_training_data(X_train, y_train, estimator, param_grid):
    """
    Tune a model on a training subset using LOOCV grid search.
    """
    n_train = len(y_train)

    if n_train < 2:
        raise ValueError("At least two training samples are required for inner LOOCV tuning.")

    inner_cv = LeaveOneOut()

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=inner_cv,
        n_jobs=N_JOBS,
        error_score=np.nan,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    if not hasattr(grid_search, "best_estimator_"):
        raise ValueError("Grid search did not return a best estimator.")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def run_nested_loocv_model(X, y, estimator, param_grid):
    """
    Run nested LOOCV.

    For each outer LOOCV split:
    1. Hold out one sample.
    2. Tune hyperparameters only on the remaining training samples.
    3. Fit the best estimator on the outer training subset.
    4. Predict the held-out sample.
    """
    outer_cv = LeaveOneOut()

    y_true_all = []
    y_pred_all = []
    fold_best_params = []
    fold_status = []

    for fold_index, (train_index, test_index) in enumerate(outer_cv.split(X), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        try:
            best_estimator, best_params, best_score = tune_model_on_training_data(
                X_train=X_train,
                y_train=y_train,
                estimator=estimator,
                param_grid=param_grid,
            )

            y_pred = best_estimator.predict(X_test)

            if hasattr(y_pred, "ravel"):
                y_pred = y_pred.ravel()

            y_true_all.append(float(y_test[0]))
            y_pred_all.append(float(y_pred[0]))

            fold_best_params.append({
                "fold": fold_index,
                "best_params": params_to_string(best_params),
                "best_inner_score": best_score,
            })

            fold_status.append("ok")

        except Exception as error:
            y_true_all.append(float(y_test[0]))
            y_pred_all.append(np.nan)

            fold_best_params.append({
                "fold": fold_index,
                "best_params": "",
                "best_inner_score": np.nan,
            })

            fold_status.append(f"fold_error: {str(error)}")

    q2, rmsecv, maecv = calculate_q2_rmsecv_maecv(y_true_all, y_pred_all)

    return {
        "Q2": q2,
        "RMSECV": rmsecv,
        "MAECV": maecv,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "fold_best_params": fold_best_params,
        "fold_status": fold_status,
    }


def run_fixed_setting_loocv_model(X, y, estimator, param_grid):
    """
    Run a simpler non-nested workflow.

    Steps:
    1. Tune hyperparameters on the full dataset using LOOCV.
    2. Use the selected setting for outer LOOCV prediction.

    This option is faster but less rigorous because the hyperparameter setting is
    selected using all samples.
    """
    best_estimator, best_params, best_score = tune_model_on_training_data(
        X_train=X,
        y_train=y,
        estimator=estimator,
        param_grid=param_grid,
    )

    outer_cv = LeaveOneOut()

    y_true_all = []
    y_pred_all = []

    for train_index, test_index in outer_cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = best_estimator
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(y_pred, "ravel"):
            y_pred = y_pred.ravel()

        y_true_all.append(float(y_test[0]))
        y_pred_all.append(float(y_pred[0]))

    q2, rmsecv, maecv = calculate_q2_rmsecv_maecv(y_true_all, y_pred_all)

    return {
        "Q2": q2,
        "RMSECV": rmsecv,
        "MAECV": maecv,
        "best_params_full_data": params_to_string(best_params),
        "best_score_full_data": best_score,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
    }


def fit_full_data_best_model(X, y, estimator, param_grid):
    """
    Tune the model on the full dataset for reporting the selected full-data
    hyperparameter setting.
    """
    try:
        best_estimator, best_params, best_score = tune_model_on_training_data(
            X_train=X,
            y_train=y,
            estimator=estimator,
            param_grid=param_grid,
        )

        return params_to_string(best_params), best_score, "ok"

    except Exception as error:
        return "", np.nan, f"full_data_tuning_error: {str(error)}"


def run_one_model(X, y, model_name, estimator, param_grid):
    """
    Run one model and return one dataset-level result row.
    """
    try:
        if USE_NESTED_LOOCV_TUNING:
            cv_result = run_nested_loocv_model(
                X=X,
                y=y,
                estimator=estimator,
                param_grid=param_grid,
            )

            full_params, full_score, full_status = fit_full_data_best_model(
                X=X,
                y=y,
                estimator=estimator,
                param_grid=param_grid,
            )

            fold_errors = [
                status for status in cv_result["fold_status"]
                if status != "ok"
            ]

            if len(fold_errors) == 0:
                status = "ok"
            else:
                status = f"partial_fold_errors: {len(fold_errors)}"

            return {
                "model": model_name,
                "Q2": cv_result["Q2"],
                "RMSECV": cv_result["RMSECV"],
                "MAECV": cv_result["MAECV"],
                "best_params_full_data": full_params,
                "best_score_full_data": full_score,
                "full_data_tuning_status": full_status,
                "nested_tuning": True,
                "status": status,
                "fold_best_params": params_to_string(cv_result["fold_best_params"]),
                "fold_status": params_to_string(cv_result["fold_status"]),
            }

        fixed_result = run_fixed_setting_loocv_model(
            X=X,
            y=y,
            estimator=estimator,
            param_grid=param_grid,
        )

        return {
            "model": model_name,
            "Q2": fixed_result["Q2"],
            "RMSECV": fixed_result["RMSECV"],
            "MAECV": fixed_result["MAECV"],
            "best_params_full_data": fixed_result["best_params_full_data"],
            "best_score_full_data": fixed_result["best_score_full_data"],
            "full_data_tuning_status": "ok",
            "nested_tuning": False,
            "status": "ok",
            "fold_best_params": "",
            "fold_status": "",
        }

    except Exception as error:
        return {
            "model": model_name,
            "Q2": np.nan,
            "RMSECV": np.nan,
            "MAECV": np.nan,
            "best_params_full_data": "",
            "best_score_full_data": np.nan,
            "full_data_tuning_status": "",
            "nested_tuning": USE_NESTED_LOOCV_TUNING,
            "status": f"model_error: {str(error)}",
            "fold_best_params": "",
            "fold_status": "",
        }


def run_one_dataset(file_path):
    """
    Run M3-M6 for one dataset.
    """
    rows = []

    dataset_id = normalize_filename(file_path.name)

    try:
        X, y = read_dataset(file_path)
        n_samples, n_features = X.shape

        eligible, reason = check_ml_eligibility(X, y)

        if not eligible:
            rows.append({
                "dataset_id": dataset_id,
                "analysis_group": ANALYSIS_GROUP_NAME,
                "model": "",
                "Q2": np.nan,
                "RMSECV": np.nan,
                "MAECV": np.nan,
                "best_params_full_data": "",
                "best_score_full_data": np.nan,
                "full_data_tuning_status": "",
                "nested_tuning": USE_NESTED_LOOCV_TUNING,
                "status": reason,
                "n_samples": n_samples,
                "n_features": n_features,
            })
            return rows

        models = get_models_and_grids(
            n_samples=n_samples,
            n_features=n_features,
        )

        for model_name in MODEL_ORDER:
            model_info = models[model_name]

            result = run_one_model(
                X=X,
                y=y,
                model_name=model_name,
                estimator=model_info["estimator"],
                param_grid=model_info["param_grid"],
            )

            rows.append({
                "dataset_id": dataset_id,
                "analysis_group": ANALYSIS_GROUP_NAME,
                "model": result["model"],
                "Q2": result["Q2"],
                "RMSECV": result["RMSECV"],
                "MAECV": result["MAECV"],
                "best_params_full_data": result["best_params_full_data"],
                "best_score_full_data": result["best_score_full_data"],
                "full_data_tuning_status": result["full_data_tuning_status"],
                "nested_tuning": result["nested_tuning"],
                "status": result["status"],
                "n_samples": n_samples,
                "n_features": n_features,
                "fold_best_params": result["fold_best_params"],
                "fold_status": result["fold_status"],
            })

        return rows

    except Exception as error:
        rows.append({
            "dataset_id": dataset_id,
            "analysis_group": ANALYSIS_GROUP_NAME,
            "model": "",
            "Q2": np.nan,
            "RMSECV": np.nan,
            "MAECV": np.nan,
            "best_params_full_data": "",
            "best_score_full_data": np.nan,
            "full_data_tuning_status": "",
            "nested_tuning": USE_NESTED_LOOCV_TUNING,
            "status": f"dataset_error: {str(error)}",
            "n_samples": np.nan,
            "n_features": np.nan,
            "fold_best_params": "",
            "fold_status": "",
        })

        return rows


# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================

def summarize_by_model(result_df):
    """
    Summarize each metric by model.
    """
    if result_df.empty:
        return pd.DataFrame()

    ok = result_df[result_df["status"].astype(str).str.startswith("ok")].copy()

    if ok.empty:
        return pd.DataFrame()

    summary_rows = []

    for metric in PREDICTIVE_METRICS:
        temp = (
            ok.groupby("model")[metric]
            .agg(
                n="count",
                mean="mean",
                sd="std",
                median="median",
                q1=lambda x: x.quantile(0.25),
                q3=lambda x: x.quantile(0.75),
                min="min",
                max="max",
            )
            .reset_index()
        )

        temp.insert(0, "metric", metric)
        temp["iqr"] = temp["q3"] - temp["q1"]

        summary_rows.append(temp)

    if not summary_rows:
        return pd.DataFrame()

    summary = pd.concat(summary_rows, ignore_index=True)

    summary["median_iqr"] = summary.apply(
        lambda row: f"{row['median']:.4g} ({row['q1']:.4g}-{row['q3']:.4g})"
        if pd.notna(row["median"]) else "",
        axis=1,
    )

    return summary


def make_metric_wide_summary(summary_df):
    """
    Convert model summary into a wide median-IQR table.
    """
    if summary_df.empty:
        return pd.DataFrame()

    wide = (
        summary_df
        .pivot_table(
            index="metric",
            columns="model",
            values="median_iqr",
            aggfunc="first",
        )
        .reset_index()
    )

    return wide


def count_best_models(result_df):
    """
    Count the best model for each dataset according to Q2, RMSECV, and MAECV.
    """
    if result_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    ok = result_df[result_df["status"].astype(str).str.startswith("ok")].copy()

    if ok.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []

    for dataset_id, sub in ok.groupby("dataset_id"):
        row = {"dataset_id": dataset_id}

        if sub["Q2"].notna().any():
            row["best_Q2_model"] = sub.loc[sub["Q2"].idxmax(), "model"]
            row["best_Q2_value"] = sub["Q2"].max()
        else:
            row["best_Q2_model"] = np.nan
            row["best_Q2_value"] = np.nan

        if sub["RMSECV"].notna().any():
            row["best_RMSECV_model"] = sub.loc[sub["RMSECV"].idxmin(), "model"]
            row["best_RMSECV_value"] = sub["RMSECV"].min()
        else:
            row["best_RMSECV_model"] = np.nan
            row["best_RMSECV_value"] = np.nan

        if sub["MAECV"].notna().any():
            row["best_MAECV_model"] = sub.loc[sub["MAECV"].idxmin(), "model"]
            row["best_MAECV_value"] = sub["MAECV"].min()
        else:
            row["best_MAECV_model"] = np.nan
            row["best_MAECV_value"] = np.nan

        rows.append(row)

    best_df = pd.DataFrame(rows)

    count_rows = []

    for col in ["best_Q2_model", "best_RMSECV_model", "best_MAECV_model"]:
        temp = best_df[col].value_counts(dropna=False).reset_index()
        temp.columns = ["model", "n"]

        temp["metric"] = col.replace("best_", "").replace("_model", "")
        temp["percent"] = temp["n"] / temp["n"].sum() * 100

        count_rows.append(temp)

    count_summary = pd.concat(count_rows, ignore_index=True)

    count_summary = count_summary[
        ["metric", "model", "n", "percent"]
    ].sort_values(["metric", "n"], ascending=[True, False])

    return best_df, count_summary


def make_status_summary(result_df):
    """
    Summarize model and dataset running status.
    """
    if result_df.empty:
        return pd.DataFrame()

    status_summary = (
        result_df
        .groupby(["model", "status"], dropna=False)
        .agg(n=("dataset_id", "count"))
        .reset_index()
        .sort_values(["model", "n"], ascending=[True, False])
    )

    return status_summary


def make_dataset_structure_summary(result_df):
    """
    Summarize basic dataset structure.
    """
    if result_df.empty:
        return pd.DataFrame()

    dataset_level = (
        result_df
        .groupby("dataset_id", dropna=False)
        .agg(
            n_samples=("n_samples", "first"),
            n_features=("n_features", "first"),
        )
        .reset_index()
    )

    structure_summary = pd.DataFrame({
        "item": [
            "n_datasets",
            "median_n_samples",
            "q1_n_samples",
            "q3_n_samples",
            "median_n_features",
            "q1_n_features",
            "q3_n_features",
        ],
        "value": [
            dataset_level["dataset_id"].nunique(),
            dataset_level["n_samples"].median(),
            dataset_level["n_samples"].quantile(0.25),
            dataset_level["n_samples"].quantile(0.75),
            dataset_level["n_features"].median(),
            dataset_level["n_features"].quantile(0.25),
            dataset_level["n_features"].quantile(0.75),
        ],
    })

    return structure_summary


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_checkpoint(all_rows, output_file, current_index):
    """
    Save temporary checkpoint results.
    """
    if not SAVE_CHECKPOINTS:
        return

    output_file = Path(output_file)
    checkpoint_file = output_file.with_name(
        f"{output_file.stem}_checkpoint_{current_index}.xlsx"
    )

    temp_df = pd.DataFrame(all_rows)
    temp_df.to_excel(checkpoint_file, index=False)


def write_output_workbook(
    result_df,
    summary_by_model,
    metric_wide_summary,
    best_model_by_dataset,
    best_model_counts,
    status_summary,
    dataset_structure_summary,
    run_log,
    output_file,
):
    """
    Write all output tables into one Excel workbook.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    run_log_df = pd.DataFrame({"log": run_log})

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        result_df.to_excel(
            writer,
            sheet_name="01_dataset_level_results",
            index=False,
        )

        if not summary_by_model.empty:
            summary_by_model.to_excel(
                writer,
                sheet_name="02_summary_by_model",
                index=False,
            )

        if not metric_wide_summary.empty:
            metric_wide_summary.to_excel(
                writer,
                sheet_name="03_metric_wide_summary",
                index=False,
            )

        if not best_model_by_dataset.empty:
            best_model_by_dataset.to_excel(
                writer,
                sheet_name="04_best_model_dataset",
                index=False,
            )

        if not best_model_counts.empty:
            best_model_counts.to_excel(
                writer,
                sheet_name="05_best_model_counts",
                index=False,
            )

        if not status_summary.empty:
            status_summary.to_excel(
                writer,
                sheet_name="06_status_summary",
                index=False,
            )

        if not dataset_structure_summary.empty:
            dataset_structure_summary.to_excel(
                writer,
                sheet_name="07_dataset_structure",
                index=False,
            )

        run_log_df.to_excel(
            writer,
            sheet_name="08_run_log",
            index=False,
        )


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """
    Run the machine-learning sensitivity analysis.
    """
    if is_placeholder_path(DATA_DIR):
        raise ValueError("Please fill DATA_DIR in the USER SETTINGS section.")

    if is_placeholder_path(OUTPUT_FILE):
        raise ValueError("Please fill OUTPUT_FILE in the USER SETTINGS section.")

    data_dir = Path(DATA_DIR)
    output_file = Path(OUTPUT_FILE)

    run_log = []

    files = get_data_files(data_dir)

    run_log.append(f"Input folder: {data_dir}")
    run_log.append(f"Number of detected dataset files: {len(files)}")
    run_log.append(f"Nested LOOCV tuning enabled: {USE_NESTED_LOOCV_TUNING}")
    run_log.append(f"Output file: {output_file}")

    print("=" * 90)
    print("Machine-learning sensitivity analysis started")
    print("=" * 90)
    print(f"Input folder: {data_dir}")
    print(f"Detected dataset files: {len(files)}")
    print(f"Nested LOOCV tuning enabled: {USE_NESTED_LOOCV_TUNING}")
    print("-" * 90)

    all_rows = []

    for index, file_path in enumerate(files, start=1):
        message = f"Running {index}/{len(files)}: {file_path.name}"
        print(message)
        run_log.append(message)

        rows = run_one_dataset(file_path)
        all_rows.extend(rows)

        if SAVE_CHECKPOINTS and CHECKPOINT_INTERVAL > 0:
            if index % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    all_rows=all_rows,
                    output_file=output_file,
                    current_index=index,
                )

                run_log.append(f"Checkpoint saved after {index} datasets.")

    result_df = pd.DataFrame(all_rows)

    summary_by_model = summarize_by_model(result_df)
    metric_wide_summary = make_metric_wide_summary(summary_by_model)

    best_model_by_dataset, best_model_counts = count_best_models(result_df)

    status_summary = make_status_summary(result_df)
    dataset_structure_summary = make_dataset_structure_summary(result_df)

    write_output_workbook(
        result_df=result_df,
        summary_by_model=summary_by_model,
        metric_wide_summary=metric_wide_summary,
        best_model_by_dataset=best_model_by_dataset,
        best_model_counts=best_model_counts,
        status_summary=status_summary,
        dataset_structure_summary=dataset_structure_summary,
        run_log=run_log,
        output_file=output_file,
    )

    print("-" * 90)
    print("Machine-learning sensitivity analysis completed")
    print(f"Output file: {output_file}")

    print("\nModel running status:")
    if not status_summary.empty:
        print(status_summary.to_string(index=False))
    else:
        print("No status summary available.")

    print("\nSummary by model:")
    if not summary_by_model.empty:
        print(summary_by_model.to_string(index=False))
    else:
        print("No model summary available.")

    print("\nBest-model counts:")
    if not best_model_counts.empty:
        print(best_model_counts.to_string(index=False))
    else:
        print("No best-model count summary available.")

    print("\nMost important output sheets:")
    print("01_dataset_level_results : dataset-level M3-M6 results")
    print("02_summary_by_model      : model-level Q2/RMSECV/MAECV summary")
    print("03_metric_wide_summary   : compact median(IQR) table")
    print("04_best_model_dataset    : best model for each dataset")
    print("05_best_model_counts     : best-model frequency counts")
    print("06_status_summary        : model running status")
    print("07_dataset_structure     : dataset structure summary")
    print("08_run_log               : run log")
    print("=" * 90)


if __name__ == "__main__":
    main()