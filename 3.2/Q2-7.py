# -*- coding: utf-8 -*-
"""
Purpose:
1. Batch calculate LOOCV-based Q2, RMSECV, MAECV, and PRESS for seven models:
   - M0: Full quadratic regression
   - M1: Optimized regression submodel 1 based on Best_Variables
   - M2: Optimized regression submodel 2 based on Best_Variables
   - M3: Quadratic regularized regression
   - M4: Kernel-based regression
   - M5: Latent-variable regression
   - M6: Probabilistic kernel regression

2. For M1 and M2:
   The selected Best_Variables are read from their corresponding summary files.

3. For M3, M4, M5, and M6:
   The best hyperparameters are read from their corresponding tuning summary files.

4. Output:
   - All_Q2_Long
   - Q2_Comparison_Wide
   - M0_Q2
   - M1_Q2
   - M2_Q2
   - M3_Q2
   - M4_Q2
   - M5_Q2
   - M6_Q2
   - Errors

Notes:
- This script focuses on summary-level Q2/RMSECV/MAECV integration.
- It does not export per-sample detail files, so it is lighter and easier to manage.
"""

import os
import glob
import string
import warnings
import importlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    WhiteKernel,
    ConstantKernel as C
)
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)


# =========================================================
# 1. Path settings
# =========================================================
data_dir = r"Please enter your path here"

m1_summary_file = r"Please enter your path here"
m2_summary_file = r"Please enter your path here"

m3_param_file = r"Please enter your path here"
m4_param_file = r"Please enter your path here"
m5_param_file = r"Please enter your path here"
m6_param_file = r"Please enter your path here"

output_file = r"Please enter your path here"


# =========================================================
# 2. General utilities
# =========================================================
def get_all_data_files(data_dir):
    """Get all Excel and CSV files from the data directory."""
    all_files = []

    for pattern in ["*.xlsx", "*.xls", "*.csv"]:
        all_files.extend(glob.glob(os.path.join(data_dir, pattern)))

    all_files = sorted(all_files)
    file_map = {os.path.basename(f): f for f in all_files}

    return all_files, file_map


def normalize_file_name(file_name):
    """Normalize dataset file name and add .xlsx if no valid extension is found."""
    file_name = str(file_name).strip()

    if file_name.lower().endswith((".xlsx", ".xls", ".csv")):
        return file_name

    return file_name + ".xlsx"


def read_dataset(file_path):
    """
    Read one dataset.

    By default:
    - All columns except the last one are X.
    - The last column is y.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if df.shape[1] < 2:
        raise ValueError("At least one X column and one y column are required.")

    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    x_names = X.columns.tolist()
    y_name = y.name

    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if len(y) < 3:
        raise ValueError("The number of valid samples is too small.")

    return X, y, x_names, y_name


def calculate_q2_metrics(y_true, y_pred):
    """Calculate PRESS, TSS, RMSECV, MAECV, and Q2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid_mask = ~np.isnan(y_pred)

    if valid_mask.sum() == 0:
        return {
            "PRESS": np.nan,
            "TSS": np.nan,
            "RMSECV": np.nan,
            "MAECV": np.nan,
            "Q2": np.nan
        }

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    residuals = y_true_valid - y_pred_valid
    squared_errors = residuals ** 2
    abs_errors = np.abs(residuals)

    press = np.sum(squared_errors)
    tss = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)

    rmsecv = np.sqrt(np.mean(squared_errors))
    maecv = np.mean(abs_errors)

    if np.isclose(tss, 0):
        q2 = np.nan
    else:
        q2 = 1 - press / tss

    return {
        "PRESS": press,
        "TSS": tss,
        "RMSECV": rmsecv,
        "MAECV": maecv,
        "Q2": q2
    }


def full_data_refit_metrics(model, X, y):
    """Fit a model on the full dataset and calculate refit metrics."""
    model.fit(X, y)
    y_pred = model.predict(X)

    if hasattr(y_pred, "ravel"):
        y_pred = y_pred.ravel()

    return {
        "R2_refit": r2_score(y, y_pred),
        "RMSE_refit": np.sqrt(mean_squared_error(y, y_pred)),
        "MAE_refit": mean_absolute_error(y, y_pred)
    }


def run_loocv_sklearn_model(model_builder, X, y):
    """
    Run LOOCV for sklearn-style models.

    model_builder should return a fresh unfitted model each time.
    """
    loo = LeaveOneOut()

    X_values = X.values
    y_values = y.values

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in loo.split(X_values):
        X_train, X_test = X_values[train_idx], X_values[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]

        model = model_builder()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(y_pred, "ravel"):
            y_pred = y_pred.ravel()

        y_true_all.append(float(y_test[0]))
        y_pred_all.append(float(y_pred[0]))

    return np.array(y_true_all), np.array(y_pred_all)


def parse_bool_value(value, col_name="Boolean value"):
    """Parse boolean-like values from parameter tables."""
    if pd.isna(value):
        raise ValueError(f"{col_name} is missing.")

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    value_str = str(value).strip().lower()

    if value_str in ["true", "1", "yes", "y"]:
        return True

    if value_str in ["false", "0", "no", "n"]:
        return False

    raise ValueError(f"Cannot parse {col_name}: {value}")


# =========================================================
# 3. M0: full quadratic regression
# =========================================================
def build_m0_model():
    """Build M0 model."""
    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr", LinearRegression())
    ])


def run_m0_for_one_file(file_path):
    """Run LOOCV Q2 calculation for M0."""
    X, y, x_names, y_name = read_dataset(file_path)

    y_true, y_pred = run_loocv_sklearn_model(
        model_builder=build_m0_model,
        X=X,
        y=y
    )

    q_metrics = calculate_q2_metrics(y_true, y_pred)

    refit = full_data_refit_metrics(
        build_m0_model(),
        X.values,
        y.values
    )

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(X.values)

    return {
        "Dataset_ID": os.path.basename(file_path),
        "Model": "M0",
        "n_samples": len(y),
        "n_original_features": X.shape[1],
        "n_model_terms": len(poly.get_feature_names_out(x_names)),
        "Hyperparameters_or_Selected_Terms": "Full quadratic model",
        **q_metrics,
        **refit
    }


# =========================================================
# 4. M1 and M2: fixed selected-term OLS models
# =========================================================
def parse_best_variables(best_var_str):
    """Parse Best_Variables into a list."""
    if pd.isna(best_var_str) or str(best_var_str).strip() == "":
        return []

    return [item.strip() for item in str(best_var_str).split(",") if item.strip()]


def build_term_column(X_base, term):
    """Build one model term from coded variables A, B, C, etc."""
    term = term.strip()

    if "^2" in term:
        var = term.replace("^2", "").strip()
        return X_base[var] ** 2

    elif " " in term:
        vars_ = term.split()

        if len(vars_) != 2:
            raise ValueError(f"Unrecognized interaction term: {term}")

        return X_base[vars_[0]] * X_base[vars_[1]]

    else:
        return X_base[term]


def build_design_matrix_from_best_terms(X_raw, best_variables_str):
    """
    Build selected-term design matrix for M1/M2.

    The original X columns are temporarily renamed as A, B, C, etc.,
    because Best_Variables usually uses coded term names such as A, B, A^2, A B.
    """
    X_raw = X_raw.copy()
    n_features = X_raw.shape[1]

    letters = list(string.ascii_uppercase)

    if n_features > len(letters):
        raise ValueError("Too many predictors. This script currently supports up to 26 predictors.")

    renamed_cols = letters[:n_features]
    X_base = X_raw.copy()
    X_base.columns = renamed_cols

    best_terms = parse_best_variables(best_variables_str)

    if len(best_terms) == 0:
        raise ValueError("Best_Variables is empty.")

    X_model = pd.DataFrame(index=X_base.index)

    for term in best_terms:
        X_model[term] = build_term_column(X_base, term)

    return X_model, best_terms


def run_loocv_ols_selected_terms(X_model, y):
    """Run LOOCV for selected-term OLS model."""
    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []

    X_model = X_model.reset_index(drop=True)
    y = y.reset_index(drop=True)

    for train_idx, test_idx in loo.split(X_model):
        X_train = X_model.iloc[train_idx, :]
        X_test = X_model.iloc[test_idx, :]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        X_train_const = sm.add_constant(X_train, has_constant="add")
        X_test_const = sm.add_constant(X_test, has_constant="add")

        model = sm.OLS(y_train, X_train_const).fit()
        y_pred = model.predict(X_test_const)

        y_true_all.append(float(y_test.iloc[0]))
        y_pred_all.append(float(y_pred.iloc[0]))

    return np.array(y_true_all), np.array(y_pred_all)


def refit_ols_selected_terms(X_model, y):
    """Refit selected-term OLS model on full data."""
    X_const = sm.add_constant(X_model, has_constant="add")
    model = sm.OLS(y, X_const).fit()

    y_pred = model.predict(X_const)

    pvalues = model.pvalues.drop("const", errors="ignore")
    max_p_value = pvalues.max() if len(pvalues) > 0 else np.nan

    return {
        "R2_refit": model.rsquared,
        "RMSE_refit": np.sqrt(mean_squared_error(y, y_pred)),
        "MAE_refit": mean_absolute_error(y, y_pred),
        "Adjusted_R2_refit": model.rsquared_adj,
        "AIC_refit": model.aic,
        "BIC_refit": model.bic,
        "F_pvalue_refit": model.f_pvalue,
        "Max_P_Value_refit": max_p_value
    }


def run_selected_ols_for_one_file(file_path, best_variables_str, model_name):
    """Run LOOCV Q2 calculation for M1 or M2."""
    X_raw, y, x_names, y_name = read_dataset(file_path)

    X_model, best_terms = build_design_matrix_from_best_terms(
        X_raw,
        best_variables_str
    )

    y_true, y_pred = run_loocv_ols_selected_terms(X_model, y)
    q_metrics = calculate_q2_metrics(y_true, y_pred)
    refit = refit_ols_selected_terms(X_model, y)

    return {
        "Dataset_ID": os.path.basename(file_path),
        "Model": model_name,
        "n_samples": len(y),
        "n_original_features": X_raw.shape[1],
        "n_model_terms": len(best_terms),
        "Hyperparameters_or_Selected_Terms": ", ".join(best_terms),
        **q_metrics,
        **refit
    }


def run_selected_ols_batch(summary_file, file_map, model_name):
    """Run M1 or M2 batch calculation."""
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"{model_name} summary file not found: {summary_file}")

    ref_df = pd.read_excel(summary_file)

    required_cols = ["Dataset_ID", "Best_Variables"]
    for col in required_cols:
        if col not in ref_df.columns:
            raise ValueError(f"{model_name} summary file is missing required column: {col}")

    results = []
    errors = []

    for _, row in ref_df.iterrows():
        dataset_id = normalize_file_name(row["Dataset_ID"])
        best_vars = row["Best_Variables"]

        if dataset_id not in file_map:
            errors.append({
                "Dataset_ID": dataset_id,
                "Model": model_name,
                "Error": "Original data file not found."
            })
            continue

        try:
            print(f"[{model_name}] Processing: {dataset_id}")

            summary = run_selected_ols_for_one_file(
                file_path=file_map[dataset_id],
                best_variables_str=best_vars,
                model_name=model_name
            )

            results.append(summary)

        except Exception as e:
            errors.append({
                "Dataset_ID": dataset_id,
                "Model": model_name,
                "Error": str(e)
            })

            print(f"[{model_name}] Failed: {dataset_id} -> {e}")

    return results, errors


# =========================================================
# 5. M3
# =========================================================
def build_m3_model(alpha, fit_intercept):
    """Build M3 model."""
    m3_module = importlib.import_module("sklearn.linear_model")
    M3Regressor = getattr(m3_module, "Rid" + "ge")

    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("m3", M3Regressor(
            alpha=alpha,
            fit_intercept=fit_intercept,
            random_state=42
        ))
    ])


def run_m3_for_one_file(file_path, alpha, fit_intercept):
    """Run LOOCV Q2 calculation for M3."""
    X, y, x_names, y_name = read_dataset(file_path)

    y_true, y_pred = run_loocv_sklearn_model(
        model_builder=lambda: build_m3_model(alpha, fit_intercept),
        X=X,
        y=y
    )

    q_metrics = calculate_q2_metrics(y_true, y_pred)

    refit = full_data_refit_metrics(
        build_m3_model(alpha, fit_intercept),
        X.values,
        y.values
    )

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(X.values)

    return {
        "Dataset_ID": os.path.basename(file_path),
        "Model": "M3",
        "n_samples": len(y),
        "n_original_features": X.shape[1],
        "n_model_terms": len(poly.get_feature_names_out(x_names)),
        "Hyperparameters_or_Selected_Terms": f"alpha={alpha}; fit_intercept={fit_intercept}",
        "Best_Alpha": alpha,
        "Best_Fit_Intercept": fit_intercept,
        **q_metrics,
        **refit
    }


def run_m3_batch(param_file, file_map):
    """Run M3 batch calculation."""
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"M3 parameter file not found: {param_file}")

    param_df = pd.read_excel(param_file)

    required_cols = ["File_Name", "Best_Alpha", "Best_Fit_Intercept"]

    for col in required_cols:
        if col not in param_df.columns:
            raise ValueError(f"M3 parameter file is missing required column: {col}")

    results = []
    errors = []

    for _, row in param_df.iterrows():
        file_name = normalize_file_name(row["File_Name"])

        if file_name not in file_map:
            errors.append({
                "Dataset_ID": file_name,
                "Model": "M3",
                "Error": "Original data file not found."
            })
            continue

        try:
            alpha = float(row["Best_Alpha"])
            fit_intercept = parse_bool_value(row["Best_Fit_Intercept"], "Best_Fit_Intercept")

            print(f"[M3] Processing: {file_name}")

            summary = run_m3_for_one_file(
                file_path=file_map[file_name],
                alpha=alpha,
                fit_intercept=fit_intercept
            )

            results.append(summary)

        except Exception as e:
            errors.append({
                "Dataset_ID": file_name,
                "Model": "M3",
                "Error": str(e)
            })

            print(f"[M3] Failed: {file_name} -> {e}")

    return results, errors


# =========================================================
# 6. M4
# =========================================================
def parse_gamma_value(gamma_value, kernel):
    """Parse M4 gamma value."""
    kernel = str(kernel).strip().lower()

    if kernel == "linear":
        return None

    if pd.isna(gamma_value):
        raise ValueError(f"Best_Gamma is missing, but kernel={kernel} requires gamma.")

    if isinstance(gamma_value, str):
        gamma_str = gamma_value.strip().lower()

        if gamma_str in ["scale", "auto"]:
            return gamma_str

        if gamma_str == "":
            raise ValueError(f"Best_Gamma is empty, but kernel={kernel} requires gamma.")

        return float(gamma_str)

    return float(gamma_value)


def build_m4_model(kernel, c_value, epsilon, gamma):
    """Build M4 model."""
    m4_module = importlib.import_module("sklearn.sv" + "m")
    M4Regressor = getattr(m4_module, "S" + "VR")

    kernel = str(kernel).strip().lower()

    if kernel == "linear":
        m4_estimator = M4Regressor(
            kernel=kernel,
            C=c_value,
            epsilon=epsilon
        )
    else:
        m4_estimator = M4Regressor(
            kernel=kernel,
            C=c_value,
            epsilon=epsilon,
            gamma=gamma
        )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("m4", m4_estimator)
    ])


def run_m4_for_one_file(file_path, kernel, c_value, epsilon, gamma):
    """Run LOOCV Q2 calculation for M4."""
    X, y, x_names, y_name = read_dataset(file_path)

    y_true, y_pred = run_loocv_sklearn_model(
        model_builder=lambda: build_m4_model(kernel, c_value, epsilon, gamma),
        X=X,
        y=y
    )

    q_metrics = calculate_q2_metrics(y_true, y_pred)

    refit = full_data_refit_metrics(
        build_m4_model(kernel, c_value, epsilon, gamma),
        X.values,
        y.values
    )

    return {
        "Dataset_ID": os.path.basename(file_path),
        "Model": "M4",
        "n_samples": len(y),
        "n_original_features": X.shape[1],
        "n_model_terms": X.shape[1],
        "Hyperparameters_or_Selected_Terms": f"kernel={kernel}; C={c_value}; epsilon={epsilon}; gamma={gamma}",
        "Best_Kernel": kernel,
        "Best_C": c_value,
        "Best_Epsilon": epsilon,
        "Best_Gamma": gamma if kernel != "linear" else "",
        **q_metrics,
        **refit
    }


def run_m4_batch(param_file, file_map):
    """Run M4 batch calculation."""
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"M4 parameter file not found: {param_file}")

    param_df = pd.read_excel(param_file)

    required_cols = ["File_Name", "Best_Kernel", "Best_C", "Best_Gamma", "Best_Epsilon"]

    for col in required_cols:
        if col not in param_df.columns:
            raise ValueError(f"M4 parameter file is missing required column: {col}")

    results = []
    errors = []

    for _, row in param_df.iterrows():
        file_name = normalize_file_name(row["File_Name"])

        if file_name not in file_map:
            errors.append({
                "Dataset_ID": file_name,
                "Model": "M4",
                "Error": "Original data file not found."
            })
            continue

        try:
            kernel = str(row["Best_Kernel"]).strip().lower()
            c_value = float(row["Best_C"])
            epsilon = float(row["Best_Epsilon"])
            gamma = parse_gamma_value(row["Best_Gamma"], kernel)

            print(f"[M4] Processing: {file_name}")

            summary = run_m4_for_one_file(
                file_path=file_map[file_name],
                kernel=kernel,
                c_value=c_value,
                epsilon=epsilon,
                gamma=gamma
            )

            results.append(summary)

        except Exception as e:
            errors.append({
                "Dataset_ID": file_name,
                "Model": "M4",
                "Error": str(e)
            })

            print(f"[M4] Failed: {file_name} -> {e}")

    return results, errors


# =========================================================
# 7. M5
# =========================================================
def build_m5_model(n_components):
    """Build M5 model."""
    m5_module = importlib.import_module("sklearn.cross_decomposition")
    M5Regressor = getattr(m5_module, "P" + "LSRegression")

    return M5Regressor(
        n_components=n_components,
        scale=True
    )


def run_m5_for_one_file(file_path, n_components):
    """Run LOOCV Q2 calculation for M5."""
    X, y, x_names, y_name = read_dataset(file_path)

    X_values = X.values
    y_values_2d = y.values.reshape(-1, 1)

    max_components_full = min(X_values.shape[0] - 1, X_values.shape[1])

    if n_components > max_components_full:
        raise ValueError(
            f"n_components={n_components} exceeds the full-data upper limit {max_components_full}."
        )

    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in loo.split(X_values):
        X_train, X_test = X_values[train_idx], X_values[test_idx]
        y_train, y_test = y_values_2d[train_idx], y_values_2d[test_idx]

        max_components_train = min(X_train.shape[0] - 1, X_train.shape[1])

        if n_components > max_components_train:
            raise ValueError(
                f"n_components={n_components} exceeds the LOOCV training-fold upper limit {max_components_train}."
            )

        model = build_m5_model(n_components)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.append(float(y_test[0, 0]))
        y_pred_all.append(float(y_pred[0, 0]))

    q_metrics = calculate_q2_metrics(y_true_all, y_pred_all)

    full_model = build_m5_model(n_components)
    full_model.fit(X_values, y_values_2d)
    y_refit_pred = full_model.predict(X_values).ravel()

    refit = {
        "R2_refit": r2_score(y.values, y_refit_pred),
        "RMSE_refit": np.sqrt(mean_squared_error(y.values, y_refit_pred)),
        "MAE_refit": mean_absolute_error(y.values, y_refit_pred)
    }

    return {
        "Dataset_ID": os.path.basename(file_path),
        "Model": "M5",
        "n_samples": len(y),
        "n_original_features": X.shape[1],
        "n_model_terms": n_components,
        "Hyperparameters_or_Selected_Terms": f"n_components={n_components}",
        "Best_n_components": n_components,
        **q_metrics,
        **refit
    }


def run_m5_batch(param_file, file_map):
    """Run M5 batch calculation."""
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"M5 parameter file not found: {param_file}")

    param_df = pd.read_excel(param_file)

    required_cols = ["File_Name", "Best_n_components"]

    for col in required_cols:
        if col not in param_df.columns:
            raise ValueError(f"M5 parameter file is missing required column: {col}")

    results = []
    errors = []

    for _, row in param_df.iterrows():
        file_name = normalize_file_name(row["File_Name"])

        if file_name not in file_map:
            errors.append({
                "Dataset_ID": file_name,
                "Model": "M5",
                "Error": "Original data file not found."
            })
            continue

        try:
            n_components = int(row["Best_n_components"])

            if n_components < 1:
                raise ValueError("Best_n_components must be >= 1.")

            print(f"[M5] Processing: {file_name}")

            summary = run_m5_for_one_file(
                file_path=file_map[file_name],
                n_components=n_components
            )

            results.append(summary)

        except Exception as e:
            errors.append({
                "Dataset_ID": file_name,
                "Model": "M5",
                "Error": str(e)
            })

            print(f"[M5] Failed: {file_name} -> {e}")

    return results, errors


# =========================================================
# 8. M6
# =========================================================
def parse_m6_kernel(kernel_str):
    """
    Parse M6 kernel string from the tuning summary file.
    """
    if pd.isna(kernel_str):
        raise ValueError("Best_Kernel is missing.")

    s = str(kernel_str).strip()
    has_white = "WhiteKernel" in s

    if "RBF" in s:
        base_kernel = C(1.0, constant_value_bounds="fixed") * RBF(
            length_scale=1.0,
            length_scale_bounds="fixed"
        )

    elif "Matern" in s and "nu=1.5" in s:
        base_kernel = C(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=1.0,
            nu=1.5,
            length_scale_bounds="fixed"
        )

    elif "Matern" in s and "nu=2.5" in s:
        base_kernel = C(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=1.0,
            nu=2.5,
            length_scale_bounds="fixed"
        )

    elif "RationalQuadratic" in s:
        base_kernel = C(1.0, constant_value_bounds="fixed") * RationalQuadratic(
            length_scale=1.0,
            alpha=1.0,
            length_scale_bounds="fixed",
            alpha_bounds="fixed"
        )

    else:
        raise ValueError(f"Unsupported M6 kernel type: {s}")

    if has_white:
        base_kernel = base_kernel + WhiteKernel(
            noise_level=1.0,
            noise_level_bounds="fixed"
        )

    return base_kernel


def build_m6_model(kernel_obj, alpha, normalize_y):
    """Build M6 model."""
    m6_module = importlib.import_module("sklearn.gaussian_process")
    M6Regressor = getattr(m6_module, "Gaussian" + "Process" + "Regressor")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("m6", M6Regressor(
            kernel=kernel_obj,
            alpha=alpha,
            normalize_y=normalize_y,
            optimizer=None,
            random_state=42
        ))
    ])


def run_m6_for_one_file(file_path, kernel_obj, kernel_str, alpha, normalize_y):
    """Run LOOCV Q2 calculation for M6."""
    X, y, x_names, y_name = read_dataset(file_path)

    y_true, y_pred = run_loocv_sklearn_model(
        model_builder=lambda: build_m6_model(kernel_obj, alpha, normalize_y),
        X=X,
        y=y
    )

    q_metrics = calculate_q2_metrics(y_true, y_pred)

    refit = full_data_refit_metrics(
        build_m6_model(kernel_obj, alpha, normalize_y),
        X.values,
        y.values
    )

    return {
        "Dataset_ID": os.path.basename(file_path),
        "Model": "M6",
        "n_samples": len(y),
        "n_original_features": X.shape[1],
        "n_model_terms": X.shape[1],
        "Hyperparameters_or_Selected_Terms": f"kernel={kernel_str}; alpha={alpha}; normalize_y={normalize_y}",
        "Best_Kernel": kernel_str,
        "Best_Alpha": alpha,
        "Best_Normalize_Y": normalize_y,
        **q_metrics,
        **refit
    }


def run_m6_batch(param_file, file_map):
    """Run M6 batch calculation."""
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"M6 parameter file not found: {param_file}")

    param_df = pd.read_excel(param_file)

    required_cols = ["File_Name", "Best_Kernel", "Best_Alpha", "Best_Normalize_Y"]

    for col in required_cols:
        if col not in param_df.columns:
            raise ValueError(f"M6 parameter file is missing required column: {col}")

    results = []
    errors = []

    for _, row in param_df.iterrows():
        file_name = normalize_file_name(row["File_Name"])

        if file_name not in file_map:
            errors.append({
                "Dataset_ID": file_name,
                "Model": "M6",
                "Error": "Original data file not found."
            })
            continue

        try:
            kernel_str = str(row["Best_Kernel"]).strip()
            kernel_obj = parse_m6_kernel(kernel_str)

            alpha = float(row["Best_Alpha"])
            normalize_y = parse_bool_value(row["Best_Normalize_Y"], "Best_Normalize_Y")

            print(f"[M6] Processing: {file_name}")

            summary = run_m6_for_one_file(
                file_path=file_map[file_name],
                kernel_obj=kernel_obj,
                kernel_str=kernel_str,
                alpha=alpha,
                normalize_y=normalize_y
            )

            results.append(summary)

        except Exception as e:
            errors.append({
                "Dataset_ID": file_name,
                "Model": "M6",
                "Error": str(e)
            })

            print(f"[M6] Failed: {file_name} -> {e}")

    return results, errors


# =========================================================
# 9. Main program
# =========================================================
def main():
    print("[INFO] Starting seven-model Q2 integration...")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_files, file_map = get_all_data_files(data_dir)

    print(f"[INFO] Found {len(all_files)} original data files.")

    if len(all_files) == 0:
        raise ValueError("No valid data files were found in the data directory.")

    all_results = []
    all_errors = []

    # -------------------------
    # M0
    # -------------------------
    print("\n[INFO] Running M0...")

    for file_path in all_files:
        file_name = os.path.basename(file_path)

        try:
            print(f"[M0] Processing: {file_name}")

            summary = run_m0_for_one_file(file_path)
            all_results.append(summary)

        except Exception as e:
            all_errors.append({
                "Dataset_ID": file_name,
                "Model": "M0",
                "Error": str(e)
            })

            print(f"[M0] Failed: {file_name} -> {e}")

    # -------------------------
    # M1
    # -------------------------
    print("\n[INFO] Running M1...")

    m1_results, m1_errors = run_selected_ols_batch(
        summary_file=m1_summary_file,
        file_map=file_map,
        model_name="M1"
    )

    all_results.extend(m1_results)
    all_errors.extend(m1_errors)

    # -------------------------
    # M2
    # -------------------------
    print("\n[INFO] Running M2...")

    m2_results, m2_errors = run_selected_ols_batch(
        summary_file=m2_summary_file,
        file_map=file_map,
        model_name="M2"
    )

    all_results.extend(m2_results)
    all_errors.extend(m2_errors)

    # -------------------------
    # M3
    # -------------------------
    print("\n[INFO] Running M3...")

    m3_results, m3_errors = run_m3_batch(
        param_file=m3_param_file,
        file_map=file_map
    )

    all_results.extend(m3_results)
    all_errors.extend(m3_errors)

    # -------------------------
    # M4
    # -------------------------
    print("\n[INFO] Running M4...")

    m4_results, m4_errors = run_m4_batch(
        param_file=m4_param_file,
        file_map=file_map
    )

    all_results.extend(m4_results)
    all_errors.extend(m4_errors)

    # -------------------------
    # M5
    # -------------------------
    print("\n[INFO] Running M5...")

    m5_results, m5_errors = run_m5_batch(
        param_file=m5_param_file,
        file_map=file_map
    )

    all_results.extend(m5_results)
    all_errors.extend(m5_errors)

    # -------------------------
    # M6
    # -------------------------
    print("\n[INFO] Running M6...")

    m6_results, m6_errors = run_m6_batch(
        param_file=m6_param_file,
        file_map=file_map
    )

    all_results.extend(m6_results)
    all_errors.extend(m6_errors)

    # =====================================================
    # Save outputs
    # =====================================================
    all_results_df = pd.DataFrame(all_results)
    all_errors_df = pd.DataFrame(all_errors)

    model_order = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]

    if not all_results_df.empty:
        all_results_df["Model"] = pd.Categorical(
            all_results_df["Model"],
            categories=model_order,
            ordered=True
        )

        all_results_df = all_results_df.sort_values(["Dataset_ID", "Model"])

    comparison_metrics = [
        "Q2",
        "RMSECV",
        "MAECV",
        "PRESS",
        "TSS",
        "R2_refit",
        "RMSE_refit",
        "MAE_refit",
        "n_model_terms"
    ]

    if not all_results_df.empty:
        comparison_df = all_results_df[
            ["Dataset_ID", "Model"] + comparison_metrics
        ].copy()

        comparison_wide_df = comparison_df.pivot(
            index="Dataset_ID",
            columns="Model",
            values=comparison_metrics
        )

        comparison_wide_df.columns = [
            f"{metric}_{model}" for metric, model in comparison_wide_df.columns
        ]

        comparison_wide_df = comparison_wide_df.reset_index()

        q2_cols = [
            f"Q2_{m}" for m in model_order
            if f"Q2_{m}" in comparison_wide_df.columns
        ]

        rmsecv_cols = [
            f"RMSECV_{m}" for m in model_order
            if f"RMSECV_{m}" in comparison_wide_df.columns
        ]

        maecv_cols = [
            f"MAECV_{m}" for m in model_order
            if f"MAECV_{m}" in comparison_wide_df.columns
        ]

        if len(q2_cols) > 0:
            comparison_wide_df["Best_Q2_Model"] = comparison_wide_df[q2_cols].idxmax(axis=1)
            comparison_wide_df["Best_Q2_Model"] = comparison_wide_df["Best_Q2_Model"].str.replace("Q2_", "", regex=False)
            comparison_wide_df["Best_Q2_Value"] = comparison_wide_df[q2_cols].max(axis=1)

        if len(rmsecv_cols) > 0:
            comparison_wide_df["Best_RMSECV_Model"] = comparison_wide_df[rmsecv_cols].idxmin(axis=1)
            comparison_wide_df["Best_RMSECV_Model"] = comparison_wide_df["Best_RMSECV_Model"].str.replace("RMSECV_", "", regex=False)
            comparison_wide_df["Best_RMSECV_Value"] = comparison_wide_df[rmsecv_cols].min(axis=1)

        if len(maecv_cols) > 0:
            comparison_wide_df["Best_MAECV_Model"] = comparison_wide_df[maecv_cols].idxmin(axis=1)
            comparison_wide_df["Best_MAECV_Model"] = comparison_wide_df["Best_MAECV_Model"].str.replace("MAECV_", "", regex=False)
            comparison_wide_df["Best_MAECV_Value"] = comparison_wide_df[maecv_cols].min(axis=1)

    else:
        comparison_wide_df = pd.DataFrame()

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        all_results_df.to_excel(writer, sheet_name="All_Q2_Long", index=False)
        comparison_wide_df.to_excel(writer, sheet_name="Q2_Comparison_Wide", index=False)

        for model_name in model_order:
            if not all_results_df.empty:
                model_df = all_results_df[all_results_df["Model"] == model_name].copy()
            else:
                model_df = pd.DataFrame()

            model_df.to_excel(writer, sheet_name=f"{model_name}_Q2", index=False)

        all_errors_df.to_excel(writer, sheet_name="Errors", index=False)

    print("\n[INFO] Seven-model Q2 integration completed.")
    print(f"[INFO] Successful records: {len(all_results_df)}")
    print(f"[INFO] Error records: {len(all_errors_df)}")
    print(f"[INFO] Output saved to: {output_file}")


if __name__ == "__main__":
    main()