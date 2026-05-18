# -*- coding: utf-8 -*-
"""
Purpose:
1. Batch calculate maximum error and maximum absolute error for seven models:
   - M0
   - M1
   - M2
   - M3
   - M4
   - M5
   - M6

2. Error definition:
   Error = Observed_Y - Predicted_Y

3. Output metrics:
   - Max_Error: maximum signed error
   - Min_Error: minimum signed error
   - Max_Abs_Error: maximum absolute error
   - Mean_Abs_Error
   - Median_Abs_Error
   - AE_Q1
   - AE_Q3
   - AE_IQR
   - R2_refit
   - RMSE_refit
   - MAE_refit

4. For M1 and M2:
   Best_Variables are read from their corresponding summary files.

5. For M3, M4, M5, and M6:
   Best hyperparameters are read from their corresponding tuning summary files.

6. Output:
   - All_Max_Error_Long
   - Max_Error_Comparison_Wide
   - M0_Max_Error
   - M1_Max_Error
   - M2_Max_Error
   - M3_Max_Error
   - M4_Max_Error
   - M5_Max_Error
   - M6_Max_Error
   - Errors
"""

import os
import glob
import string
import warnings
import importlib
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    WhiteKernel,
    ConstantKernel
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

    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if len(y) < 3:
        raise ValueError("The number of valid samples is too small.")

    return X, y


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


def calculate_error_metrics(y_true, y_pred):
    """
    Calculate maximum error and maximum absolute error.

    Error = Observed_Y - Predicted_Y
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if hasattr(y_pred, "ravel"):
        y_pred = y_pred.ravel()

    residuals = y_true - y_pred
    abs_error = np.abs(residuals)

    if len(residuals) == 0:
        return {
            "Max_Error": np.nan,
            "Min_Error": np.nan,
            "Max_Abs_Error": np.nan,
            "Mean_Error": np.nan,
            "Median_Error": np.nan,
            "Mean_Abs_Error": np.nan,
            "Median_Abs_Error": np.nan,
            "AE_Q1": np.nan,
            "AE_Q3": np.nan,
            "AE_IQR": np.nan,
            "R2_refit": np.nan,
            "RMSE_refit": np.nan,
            "MAE_refit": np.nan
        }

    ae_q1 = np.percentile(abs_error, 25)
    ae_q3 = np.percentile(abs_error, 75)
    ae_iqr = ae_q3 - ae_q1

    return {
        "Max_Error": np.max(residuals),
        "Min_Error": np.min(residuals),
        "Max_Abs_Error": np.max(abs_error),
        "Mean_Error": np.mean(residuals),
        "Median_Error": np.median(residuals),
        "Mean_Abs_Error": np.mean(abs_error),
        "Median_Abs_Error": np.median(abs_error),
        "AE_Q1": ae_q1,
        "AE_Q3": ae_q3,
        "AE_IQR": ae_iqr,
        "R2_refit": r2_score(y_true, y_pred),
        "RMSE_refit": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE_refit": mean_absolute_error(y_true, y_pred)
    }


# =========================================================
# 3. M0
# =========================================================
def build_m0_design_matrix(X_raw):
    """Build full quadratic design matrix for M0."""
    X_full = pd.DataFrame(index=X_raw.index)
    cols = list(X_raw.columns)

    for col in cols:
        X_full[col] = X_raw[col]

    for col in cols:
        X_full[f"{col}^2"] = X_raw[col] ** 2

    for c1, c2 in itertools.combinations(cols, 2):
        X_full[f"{c1} {c2}"] = X_raw[c1] * X_raw[c2]

    return X_full


def run_m0_for_one_file(file_path):
    """Run maximum error calculation for M0."""
    X_raw, y = read_dataset(file_path)

    X_model = build_m0_design_matrix(X_raw)
    X_const = sm.add_constant(X_model, has_constant="add")

    if len(X_const) <= X_const.shape[1]:
        raise ValueError(
            f"Insufficient sample size for M0: n={len(X_const)}, parameters={X_const.shape[1]}"
        )

    model = sm.OLS(y, X_const).fit()
    y_pred = model.predict(X_const)

    metrics = calculate_error_metrics(y.values, y_pred.values)

    return {
        "Dataset_ID": os.path.basename(file_path),
        "Model": "M0",
        "n_used": len(y),
        "Nt": X_const.shape[1] - 1,
        "Adjusted_R2": model.rsquared_adj,
        **metrics
    }


# =========================================================
# 4. M1 and M2
# =========================================================
def parse_best_variables(text):
    """Parse Best_Variables into a list."""
    if pd.isna(text) or str(text).strip() == "":
        return []

    return [x.strip() for x in str(text).split(",") if x.strip()]


def build_selected_design_matrix_with_columns(X_base, terms):
    """Build selected design matrix using the current column names."""
    X = pd.DataFrame(index=X_base.index)

    for term in terms:
        if "^2" in term:
            base_var = term.replace("^2", "").strip()

            if base_var not in X_base.columns:
                raise ValueError(f"Base variable for squared term not found: {base_var}")

            X[term] = X_base[base_var] ** 2

        elif " " in term:
            parts = term.split()

            if len(parts) != 2:
                raise ValueError(f"Unrecognized interaction term format: {term}")

            v1, v2 = parts

            if v1 not in X_base.columns or v2 not in X_base.columns:
                raise ValueError(f"Variables for interaction term not found: {term}")

            X[term] = X_base[v1] * X_base[v2]

        else:
            if term not in X_base.columns:
                raise ValueError(f"Linear variable not found: {term}")

            X[term] = X_base[term]

    return X


def build_selected_design_matrix(X_raw, best_variables_text):
    """
    Build selected-term design matrix for M1/M2.

    The function first tries to use the original column names.
    If that fails, it renames columns as A, B, C, ... and tries again.
    """
    terms = parse_best_variables(best_variables_text)

    if len(terms) == 0:
        raise ValueError("Best_Variables is empty.")

    try:
        X_model = build_selected_design_matrix_with_columns(X_raw, terms)
        return X_model, terms
    except Exception:
        pass

    n_features = X_raw.shape[1]
    letters = list(string.ascii_uppercase)

    if n_features > len(letters):
        raise ValueError("Too many predictors. This script currently supports up to 26 predictors.")

    X_base = X_raw.copy()
    X_base.columns = letters[:n_features]

    X_model = build_selected_design_matrix_with_columns(X_base, terms)

    return X_model, terms


def run_selected_ols_for_one_file(file_path, best_variables_text, model_name):
    """Run maximum error calculation for M1 or M2."""
    X_raw, y = read_dataset(file_path)

    X_model, terms = build_selected_design_matrix(X_raw, best_variables_text)
    X_const = sm.add_constant(X_model, has_constant="add")

    if len(X_const) <= X_const.shape[1]:
        raise ValueError(
            f"Insufficient sample size for {model_name}: n={len(X_const)}, parameters={X_const.shape[1]}"
        )

    model = sm.OLS(y, X_const).fit()
    y_pred = model.predict(X_const)

    metrics = calculate_error_metrics(y.values, y_pred.values)

    return {
        "Dataset_ID": os.path.basename(file_path),
        "Model": model_name,
        "Best_Variables": ", ".join(terms),
        "n_used": len(y),
        "Nt": X_const.shape[1] - 1,
        "Adjusted_R2": model.rsquared_adj,
        **metrics
    }


def run_selected_ols_batch(summary_file, file_map, model_name):
    """Run maximum error calculation for M1 or M2."""
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"{model_name} summary file not found: {summary_file}")

    summary_df = pd.read_excel(summary_file)

    required_cols = ["Dataset_ID", "Best_Variables"]

    for col in required_cols:
        if col not in summary_df.columns:
            raise ValueError(f"{model_name} summary file is missing required column: {col}")

    results = []
    errors = []

    for _, row in summary_df.iterrows():
        file_name = normalize_file_name(row["Dataset_ID"])
        best_variables = row["Best_Variables"]

        if file_name not in file_map:
            errors.append({
                "Dataset_ID": file_name,
                "Model": model_name,
                "Best_Variables": best_variables,
                "Error": "Original data file not found."
            })
            continue

        try:
            print(f"[{model_name}] Processing: {file_name}")

            result = run_selected_ols_for_one_file(
                file_path=file_map[file_name],
                best_variables_text=best_variables,
                model_name=model_name
            )

            results.append(result)

        except Exception as e:
            errors.append({
                "Dataset_ID": file_name,
                "Model": model_name,
                "Best_Variables": best_variables,
                "Error": str(e)
            })

            print(f"[{model_name}] Failed: {file_name} -> {e}")

    return results, errors


# =========================================================
# 5. M3
# =========================================================
def build_m3_model(alpha, fit_intercept=True):
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


def run_m3_for_one_file(file_path, alpha, fit_intercept=True):
    """Run maximum error calculation for M3."""
    X, y = read_dataset(file_path)

    model = build_m3_model(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(X, y)

    y_pred = model.predict(X)
    metrics = calculate_error_metrics(y.values, y_pred)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(X.values)

    return {
        "Dataset_ID": os.path.basename(file_path),
        "Model": "M3",
        "n_used": len(y),
        "Nt": len(poly.get_feature_names_out(X.columns)),
        "Best_Alpha": alpha,
        "Best_Fit_Intercept": fit_intercept,
        **metrics
    }


def run_m3_batch(param_file, file_map):
    """Run maximum error calculation for M3."""
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"M3 parameter file not found: {param_file}")

    param_df = pd.read_excel(param_file)

    required_cols = ["File_Name", "Best_Alpha"]

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

            if "Best_Fit_Intercept" in param_df.columns:
                fit_intercept = parse_bool_value(row["Best_Fit_Intercept"], "Best_Fit_Intercept")
            else:
                fit_intercept = True

            print(f"[M3] Processing: {file_name}")

            result = run_m3_for_one_file(
                file_path=file_map[file_name],
                alpha=alpha,
                fit_intercept=fit_intercept
            )

            results.append(result)

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
def parse_gamma_value(value, kernel):
    """Parse M4 gamma value."""
    kernel = str(kernel).strip().lower()

    if kernel == "linear":
        return None

    if pd.isna(value):
        return "scale"

    if isinstance(value, str):
        value_str = value.strip().lower()

        if value_str in ["scale", "auto"]:
            return value_str

        if value_str == "":
            return "scale"

        return float(value_str)

    return float(value)


def build_m4_model(kernel, c_value, epsilon, gamma):
    """Build M4 model."""
    m4_module = importlib.import_module("sklearn.sv" + "m")
    M4Regressor = getattr(m4_module, "S" + "VR")

    kernel = str(kernel).strip().lower()

    if kernel == "linear":
        estimator = M4Regressor(
            kernel=kernel,
            C=c_value,
            epsilon=epsilon
        )
    else:
        estimator = M4Regressor(
            kernel=kernel,
            C=c_value,
            epsilon=epsilon,
            gamma=gamma
        )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("m4", estimator)
    ])


def run_m4_for_one_file(file_path, kernel, c_value, epsilon, gamma):
    """Run maximum error calculation for M4."""
    X, y = read_dataset(file_path)

    model = build_m4_model(
        kernel=kernel,
        c_value=c_value,
        epsilon=epsilon,
        gamma=gamma
    )

    model.fit(X, y)

    y_pred = model.predict(X)
    metrics = calculate_error_metrics(y.values, y_pred)

    return {
        "Dataset_ID": os.path.basename(file_path),
        "Model": "M4",
        "n_used": len(y),
        "Nt": X.shape[1],
        "Best_Kernel": kernel,
        "Best_C": c_value,
        "Best_Gamma": gamma if kernel != "linear" else "",
        "Best_Epsilon": epsilon,
        **metrics
    }


def run_m4_batch(param_file, file_map):
    """Run maximum error calculation for M4."""
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"M4 parameter file not found: {param_file}")

    param_df = pd.read_excel(param_file)

    required_cols = ["File_Name", "Best_Kernel", "Best_C", "Best_Epsilon"]

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

            if "Best_Gamma" in param_df.columns:
                gamma = parse_gamma_value(row["Best_Gamma"], kernel)
            else:
                gamma = None if kernel == "linear" else "scale"

            print(f"[M4] Processing: {file_name}")

            result = run_m4_for_one_file(
                file_path=file_map[file_name],
                kernel=kernel,
                c_value=c_value,
                epsilon=epsilon,
                gamma=gamma
            )

            results.append(result)

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

    return Pipeline([
        ("scaler", StandardScaler()),
        ("m5", M5Regressor(n_components=n_components))
    ])


def run_m5_for_one_file(file_path, n_components):
    """Run maximum error calculation for M5."""
    X, y = read_dataset(file_path)

    max_components = min(X.shape[1], X.shape[0] - 1)

    if n_components < 1 or n_components > max_components:
        raise ValueError(
            f"Best_n_components={n_components} is outside the allowed range 1-{max_components}."
        )

    model = build_m5_model(n_components=n_components)
    model.fit(X, y)

    y_pred = model.predict(X).ravel()
    metrics = calculate_error_metrics(y.values, y_pred)

    return {
        "Dataset_ID": os.path.basename(file_path),
        "Model": "M5",
        "n_used": len(y),
        "Nt": n_components,
        "Best_n_components": n_components,
        **metrics
    }


def run_m5_batch(param_file, file_map):
    """Run maximum error calculation for M5."""
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

            print(f"[M5] Processing: {file_name}")

            result = run_m5_for_one_file(
                file_path=file_map[file_name],
                n_components=n_components
            )

            results.append(result)

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
    """Parse M6 kernel string from the parameter table."""
    if pd.isna(kernel_str):
        raise ValueError("Best_Kernel is missing.")

    kernel_str = str(kernel_str).strip()

    safe_dict = {
        "RBF": RBF,
        "Matern": Matern,
        "RationalQuadratic": RationalQuadratic,
        "WhiteKernel": WhiteKernel,
        "ConstantKernel": ConstantKernel
    }

    try:
        return eval(kernel_str, {"__builtins__": {}}, safe_dict)
    except Exception:
        pass

    try:
        fixed_str = kernel_str.replace("1**2", "ConstantKernel(1.0)")
        return eval(fixed_str, {"__builtins__": {}}, safe_dict)
    except Exception:
        pass

    has_white = "WhiteKernel" in kernel_str

    if "RBF" in kernel_str:
        base_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    elif "Matern" in kernel_str and "nu=1.5" in kernel_str:
        base_kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
    elif "Matern" in kernel_str and "nu=2.5" in kernel_str:
        base_kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    elif "RationalQuadratic" in kernel_str:
        base_kernel = ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)
    else:
        raise ValueError(f"Unsupported M6 kernel type: {kernel_str}")

    if has_white:
        base_kernel = base_kernel + WhiteKernel(noise_level=1.0)

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
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=2,
            random_state=42
        ))
    ])


def run_m6_for_one_file(file_path, kernel_obj, kernel_str, alpha, normalize_y):
    """Run maximum error calculation for M6."""
    X, y = read_dataset(file_path)

    model = build_m6_model(
        kernel_obj=kernel_obj,
        alpha=alpha,
        normalize_y=normalize_y
    )

    model.fit(X, y)

    y_pred = model.predict(X)
    metrics = calculate_error_metrics(y.values, y_pred)

    return {
        "Dataset_ID": os.path.basename(file_path),
        "Model": "M6",
        "n_used": len(y),
        "Nt": X.shape[1],
        "Best_Kernel": kernel_str,
        "Best_Alpha": alpha,
        "Best_Normalize_Y": normalize_y,
        **metrics
    }


def run_m6_batch(param_file, file_map):
    """Run maximum error calculation for M6."""
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

            result = run_m6_for_one_file(
                file_path=file_map[file_name],
                kernel_obj=kernel_obj,
                kernel_str=kernel_str,
                alpha=alpha,
                normalize_y=normalize_y
            )

            results.append(result)

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
    print("[INFO] Starting seven-model maximum error integration...")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_files, file_map = get_all_data_files(data_dir)

    print(f"[INFO] Found {len(all_files)} original data files.")

    if len(all_files) == 0:
        raise ValueError("No valid data files were found in the data directory.")

    all_results = []
    all_errors = []

    # M0
    print("\n[INFO] Running M0...")
    for file_path in all_files:
        file_name = os.path.basename(file_path)

        try:
            print(f"[M0] Processing: {file_name}")
            result = run_m0_for_one_file(file_path)
            all_results.append(result)

        except Exception as e:
            all_errors.append({
                "Dataset_ID": file_name,
                "Model": "M0",
                "Error": str(e)
            })

            print(f"[M0] Failed: {file_name} -> {e}")

    # M1
    print("\n[INFO] Running M1...")
    m1_results, m1_errors = run_selected_ols_batch(
        summary_file=m1_summary_file,
        file_map=file_map,
        model_name="M1"
    )
    all_results.extend(m1_results)
    all_errors.extend(m1_errors)

    # M2
    print("\n[INFO] Running M2...")
    m2_results, m2_errors = run_selected_ols_batch(
        summary_file=m2_summary_file,
        file_map=file_map,
        model_name="M2"
    )
    all_results.extend(m2_results)
    all_errors.extend(m2_errors)

    # M3
    print("\n[INFO] Running M3...")
    m3_results, m3_errors = run_m3_batch(
        param_file=m3_param_file,
        file_map=file_map
    )
    all_results.extend(m3_results)
    all_errors.extend(m3_errors)

    # M4
    print("\n[INFO] Running M4...")
    m4_results, m4_errors = run_m4_batch(
        param_file=m4_param_file,
        file_map=file_map
    )
    all_results.extend(m4_results)
    all_errors.extend(m4_errors)

    # M5
    print("\n[INFO] Running M5...")
    m5_results, m5_errors = run_m5_batch(
        param_file=m5_param_file,
        file_map=file_map
    )
    all_results.extend(m5_results)
    all_errors.extend(m5_errors)

    # M6
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
        "Max_Error",
        "Min_Error",
        "Max_Abs_Error",
        "Mean_Error",
        "Median_Error",
        "Mean_Abs_Error",
        "Median_Abs_Error",
        "AE_Q1",
        "AE_Q3",
        "AE_IQR",
        "R2_refit",
        "RMSE_refit",
        "MAE_refit",
        "Adjusted_R2",
        "Nt",
        "n_used"
    ]

    existing_comparison_metrics = [
        col for col in comparison_metrics
        if not all_results_df.empty and col in all_results_df.columns
    ]

    if not all_results_df.empty:
        comparison_df = all_results_df[
            ["Dataset_ID", "Model"] + existing_comparison_metrics
        ].copy()

        comparison_wide_df = comparison_df.pivot(
            index="Dataset_ID",
            columns="Model",
            values=existing_comparison_metrics
        )

        comparison_wide_df.columns = [
            f"{metric}_{model}" for metric, model in comparison_wide_df.columns
        ]

        comparison_wide_df = comparison_wide_df.reset_index()

        max_abs_cols = [
            f"Max_Abs_Error_{m}" for m in model_order
            if f"Max_Abs_Error_{m}" in comparison_wide_df.columns
        ]

        if len(max_abs_cols) > 0:
            comparison_wide_df["Lowest_Max_Abs_Error_Model"] = comparison_wide_df[max_abs_cols].idxmin(axis=1)
            comparison_wide_df["Lowest_Max_Abs_Error_Model"] = (
                comparison_wide_df["Lowest_Max_Abs_Error_Model"]
                .str.replace("Max_Abs_Error_", "", regex=False)
            )
            comparison_wide_df["Lowest_Max_Abs_Error_Value"] = comparison_wide_df[max_abs_cols].min(axis=1)

    else:
        comparison_wide_df = pd.DataFrame()

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        all_results_df.to_excel(writer, sheet_name="All_Max_Error_Long", index=False)
        comparison_wide_df.to_excel(writer, sheet_name="Max_Error_Comparison_Wide", index=False)

        for model_name in model_order:
            if not all_results_df.empty:
                model_df = all_results_df[all_results_df["Model"] == model_name].copy()
            else:
                model_df = pd.DataFrame()

            model_df.to_excel(writer, sheet_name=f"{model_name}_Max_Error", index=False)

        all_errors_df.to_excel(writer, sheet_name="Errors", index=False)

    print("\n[INFO] Seven-model maximum error integration completed.")
    print(f"[INFO] Successful records: {len(all_results_df)}")
    print(f"[INFO] Error records: {len(all_errors_df)}")
    print(f"[INFO] Output saved to: {output_file}")


if __name__ == "__main__":
    main()