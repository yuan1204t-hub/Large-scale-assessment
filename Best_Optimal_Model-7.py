# -*- coding: utf-8 -*-
"""
Purpose:
1. Batch search optimal extraction conditions for seven models:
   - M0
   - M1
   - M2
   - M3
   - M4
   - M5
   - M6

2. For M0:
   A full quadratic model is fitted and searched.

3. For M1 and M2:
   Best_Variables are read from their corresponding summary files.

4. For M3, M4, M5, and M6:
   Best hyperparameters are read from their corresponding tuning summary files.

5. Search strategy:
   - If USE_UNIQUE_LEVELS = True:
     The search grid uses the actual experimental levels observed in each factor.
   - If USE_UNIQUE_LEVELS = False:
     The search grid uses equally spaced values between the observed minimum and maximum.

6. Output:
   - All_Optimal_Long
   - Optimal_Comparison_Wide
   - M0_Optimal
   - M1_Optimal
   - M2_Optimal
   - M3_Optimal
   - M4_Optimal
   - M5_Optimal
   - M6_Optimal
   - Errors

Notes:
- Model names are reported only as M0-M6.
- The response is optimized within the observed factor range.
- By default, the script searches for the maximum predicted response.
"""

import os
import glob
import re
import ast
import string
import warnings
import itertools
import importlib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
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
# 2. Global optimization settings
# =========================================================
OPTIMIZATION_MODE = "max"       # "max" or "min"
USE_UNIQUE_LEVELS = True        # True = use observed levels; False = use evenly spaced grid
GRID_POINTS_PER_VAR = 50        # Used only when USE_UNIQUE_LEVELS = False
CHUNK_SIZE = 200000             # Chunk size for grid prediction


# =========================================================
# 3. General utilities
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

    y_name = df.columns[-1]

    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if len(y) < 3:
        raise ValueError("The number of valid samples is too small.")

    return X, y, y_name


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


def generate_search_value_lists(X_df, use_unique_levels=True, grid_points=50):
    """Generate value lists for grid search."""
    value_lists = []

    for col in X_df.columns:
        col_values = pd.to_numeric(X_df[col], errors="coerce").dropna().values

        if len(col_values) == 0:
            raise ValueError(f"No valid values found for variable: {col}")

        if use_unique_levels:
            values = np.sort(np.unique(col_values))
        else:
            vmin = np.min(col_values)
            vmax = np.max(col_values)

            if np.isclose(vmin, vmax):
                values = np.array([vmin])
            else:
                values = np.linspace(vmin, vmax, grid_points)

        value_lists.append(values)

    return value_lists


def search_optimal_condition_with_model(
    model,
    X_search_base,
    mode="max",
    use_unique_levels=True,
    grid_points=50,
    chunk_size=200000
):
    """
    Search optimal condition for sklearn-style models.

    The model must accept a DataFrame with the same columns as X_search_base.
    """
    value_lists = generate_search_value_lists(
        X_search_base,
        use_unique_levels=use_unique_levels,
        grid_points=grid_points
    )

    total_grid_size = int(np.prod([len(v) for v in value_lists]))

    if mode == "max":
        best_pred = -np.inf
        is_better = lambda current, best: current > best
    elif mode == "min":
        best_pred = np.inf
        is_better = lambda current, best: current < best
    else:
        raise ValueError("OPTIMIZATION_MODE must be either 'max' or 'min'.")

    best_condition = None
    grid_iterator = itertools.product(*value_lists)

    while True:
        chunk = list(itertools.islice(grid_iterator, chunk_size))

        if not chunk:
            break

        chunk_df = pd.DataFrame(chunk, columns=X_search_base.columns)
        preds = model.predict(chunk_df)

        if hasattr(preds, "ravel"):
            preds = preds.ravel()

        if mode == "max":
            local_idx = int(np.argmax(preds))
        else:
            local_idx = int(np.argmin(preds))

        local_pred = float(preds[local_idx])

        if is_better(local_pred, best_pred):
            best_pred = local_pred
            best_condition = chunk_df.iloc[local_idx].to_dict()

    if best_condition is None:
        raise ValueError("Grid search failed to find an optimal condition.")

    return best_condition, best_pred, total_grid_size


def search_optimal_condition_with_custom_predictor(
    predictor_func,
    X_search_base,
    mode="max",
    use_unique_levels=True,
    grid_points=50,
    chunk_size=200000
):
    """
    Search optimal condition using a custom prediction function.

    predictor_func must accept a DataFrame and return predicted values.
    """
    value_lists = generate_search_value_lists(
        X_search_base,
        use_unique_levels=use_unique_levels,
        grid_points=grid_points
    )

    total_grid_size = int(np.prod([len(v) for v in value_lists]))

    if mode == "max":
        best_pred = -np.inf
        is_better = lambda current, best: current > best
    elif mode == "min":
        best_pred = np.inf
        is_better = lambda current, best: current < best
    else:
        raise ValueError("OPTIMIZATION_MODE must be either 'max' or 'min'.")

    best_condition = None
    grid_iterator = itertools.product(*value_lists)

    while True:
        chunk = list(itertools.islice(grid_iterator, chunk_size))

        if not chunk:
            break

        chunk_df = pd.DataFrame(chunk, columns=X_search_base.columns)
        preds = predictor_func(chunk_df)

        if hasattr(preds, "ravel"):
            preds = preds.ravel()

        if mode == "max":
            local_idx = int(np.argmax(preds))
        else:
            local_idx = int(np.argmin(preds))

        local_pred = float(preds[local_idx])

        if is_better(local_pred, best_pred):
            best_pred = local_pred
            best_condition = chunk_df.iloc[local_idx].to_dict()

    if best_condition is None:
        raise ValueError("Grid search failed to find an optimal condition.")

    return best_condition, best_pred, total_grid_size


def create_result_row(
    file_name,
    model_name,
    y_name,
    optimization_mode,
    best_pred,
    best_condition,
    grid_size,
    observed_max,
    observed_min,
    extra_info=None
):
    """Create one standardized result row."""
    result = {
        "Dataset_ID": file_name,
        "Model": model_name,
        "Y_Name": y_name,
        "Optimization_Mode": optimization_mode,
        "Best_Predicted_Y": best_pred,
        "Observed_Max_Y": observed_max,
        "Observed_Min_Y": observed_min,
        "Grid_Size": grid_size
    }

    if extra_info:
        result.update(extra_info)

    for k, v in best_condition.items():
        result[f"Optimal_{k}"] = v

    return result


# =========================================================
# 4. Term parsing utilities for M1 and M2
# =========================================================
def parse_combo(x):
    """Parse Best_Variables."""
    if pd.isna(x):
        return []

    s = str(x).strip()

    if s.startswith("[") and s.endswith("]"):
        try:
            vals = ast.literal_eval(s)
            return [str(v).strip() for v in vals if str(v).strip()]
        except Exception:
            pass

    s = s.replace("，", ",")
    return [v.strip() for v in s.split(",") if v.strip()]


def normalize_term(term):
    """
    Normalize a term for matching.

    Examples:
    - X1 X2 and X2 X1 are treated as the same interaction.
    - X1^2 is treated as X1 X1.
    """
    term = str(term).strip()

    if "^2" in term:
        base = term.replace("^2", "").strip()
        return tuple(sorted([base, base]))

    parts = [
        p.strip()
        for p in re.split(r"[^a-zA-Z0-9_]+", term)
        if p.strip()
    ]

    return tuple(sorted(parts))


def get_used_vars_from_terms(best_terms, raw_vars):
    """Extract base variables used in selected terms."""
    base_candidates = set()

    for term in best_terms:
        term = str(term).strip()

        if "^2" in term:
            base = term.replace("^2", "").strip()
            base_candidates.add(base)
        else:
            parts = [
                p.strip()
                for p in re.split(r"[^a-zA-Z0-9_]+", term)
                if p.strip()
            ]
            base_candidates.update(parts)

    used_vars = [v for v in raw_vars if v in base_candidates]

    if used_vars:
        return used_vars

    # Fallback: try coded names A, B, C...
    letters = list(string.ascii_uppercase)
    raw_to_code = {
        raw_var: letters[i]
        for i, raw_var in enumerate(raw_vars)
        if i < len(letters)
    }

    code_to_raw = {v: k for k, v in raw_to_code.items()}
    used_vars = []

    for candidate in base_candidates:
        if candidate in code_to_raw:
            used_vars.append(code_to_raw[candidate])

    used_vars = [v for v in raw_vars if v in used_vars]

    if not used_vars:
        raise ValueError("No base variables in Best_Variables matched raw data columns.")

    return used_vars


def match_selected_polynomial_terms(poly_feature_names, best_terms):
    """Match selected terms to polynomial feature indices."""
    selected_idx = []

    for selected_term in best_terms:
        norm_selected = normalize_term(selected_term)
        found = False

        for idx, feature_name in enumerate(poly_feature_names):
            if normalize_term(feature_name) == norm_selected:
                if idx not in selected_idx:
                    selected_idx.append(idx)
                found = True
                break

        if not found:
            raise ValueError(f"Selected term was not matched: {selected_term}")

    if not selected_idx:
        raise ValueError("No selected polynomial terms were matched.")

    return selected_idx


def run_selected_subset_model_for_one_file(file_path, best_variables_text, model_name):
    """Run optimal-condition search for M1 or M2."""
    X_all, y, y_name = read_dataset(file_path)

    raw_vars = list(X_all.columns)
    best_terms = parse_combo(best_variables_text)

    if len(best_terms) == 0:
        raise ValueError("Best_Variables is empty.")

    used_vars = get_used_vars_from_terms(best_terms, raw_vars)

    X_used = X_all[used_vars].copy()

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(X_used)

    all_features = poly.get_feature_names_out(used_vars)
    selected_idx = match_selected_polynomial_terms(all_features, best_terms)

    X_train_poly = poly.transform(X_used)[:, selected_idx]
    model = LinearRegression()
    model.fit(X_train_poly, y)

    def predictor(grid_df):
        grid_poly = poly.transform(grid_df)[:, selected_idx]
        return model.predict(grid_poly)

    best_condition, best_pred, grid_size = search_optimal_condition_with_custom_predictor(
        predictor_func=predictor,
        X_search_base=X_used,
        mode=OPTIMIZATION_MODE,
        use_unique_levels=USE_UNIQUE_LEVELS,
        grid_points=GRID_POINTS_PER_VAR,
        chunk_size=CHUNK_SIZE
    )

    extra_info = {
        "Best_Variables": ", ".join(best_terms),
        "Used_Variables": ", ".join(used_vars),
        "n_used_vars": len(used_vars),
        "n_selected_terms": len(best_terms)
    }

    return create_result_row(
        file_name=os.path.basename(file_path),
        model_name=model_name,
        y_name=y_name,
        optimization_mode=OPTIMIZATION_MODE,
        best_pred=best_pred,
        best_condition=best_condition,
        grid_size=grid_size,
        observed_max=float(np.max(y)),
        observed_min=float(np.min(y)),
        extra_info=extra_info
    )


def run_selected_subset_batch(summary_file, file_map, model_name):
    """Run M1 or M2 optimal-condition search."""
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
                "Error": "Original data file not found."
            })
            continue

        try:
            print(f"[{model_name}] Processing: {file_name}")

            result = run_selected_subset_model_for_one_file(
                file_path=file_map[file_name],
                best_variables_text=best_variables,
                model_name=model_name
            )

            results.append(result)

        except Exception as e:
            errors.append({
                "Dataset_ID": file_name,
                "Model": model_name,
                "Error": str(e)
            })

            print(f"[{model_name}] Failed: {file_name} -> {e}")

    return results, errors


# =========================================================
# 5. M0
# =========================================================
def run_m0_for_one_file(file_path):
    """Run optimal-condition search for M0."""
    X, y, y_name = read_dataset(file_path)

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr", LinearRegression())
    ])

    model.fit(X, y)

    best_condition, best_pred, grid_size = search_optimal_condition_with_model(
        model=model,
        X_search_base=X,
        mode=OPTIMIZATION_MODE,
        use_unique_levels=USE_UNIQUE_LEVELS,
        grid_points=GRID_POINTS_PER_VAR,
        chunk_size=CHUNK_SIZE
    )

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(X)

    extra_info = {
        "n_used_vars": X.shape[1],
        "n_selected_terms": len(poly.get_feature_names_out(X.columns)),
        "Best_Variables": "Full quadratic model",
        "Used_Variables": ", ".join(X.columns)
    }

    return create_result_row(
        file_name=os.path.basename(file_path),
        model_name="M0",
        y_name=y_name,
        optimization_mode=OPTIMIZATION_MODE,
        best_pred=best_pred,
        best_condition=best_condition,
        grid_size=grid_size,
        observed_max=float(np.max(y)),
        observed_min=float(np.min(y)),
        extra_info=extra_info
    )


# =========================================================
# 6. M3
# =========================================================
def build_m3_model(alpha, fit_intercept):
    """Build M3 model."""
    m3_module = importlib.import_module("sklearn.linear_model")
    M3Regressor = getattr(m3_module, "Rid" + "ge")

    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("m3", M3Regressor(
            alpha=float(alpha),
            fit_intercept=parse_bool_value(fit_intercept, "Best_Fit_Intercept")
        ))
    ])


def run_m3_for_one_file(file_path, alpha, fit_intercept):
    """Run optimal-condition search for M3."""
    X, y, y_name = read_dataset(file_path)

    model = build_m3_model(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(X, y)

    best_condition, best_pred, grid_size = search_optimal_condition_with_model(
        model=model,
        X_search_base=X,
        mode=OPTIMIZATION_MODE,
        use_unique_levels=USE_UNIQUE_LEVELS,
        grid_points=GRID_POINTS_PER_VAR,
        chunk_size=CHUNK_SIZE
    )

    extra_info = {
        "Best_Alpha": alpha,
        "Best_Fit_Intercept": fit_intercept,
        "n_used_vars": X.shape[1],
        "Used_Variables": ", ".join(X.columns)
    }

    return create_result_row(
        file_name=os.path.basename(file_path),
        model_name="M3",
        y_name=y_name,
        optimization_mode=OPTIMIZATION_MODE,
        best_pred=best_pred,
        best_condition=best_condition,
        grid_size=grid_size,
        observed_max=float(np.max(y)),
        observed_min=float(np.min(y)),
        extra_info=extra_info
    )


def run_m3_batch(param_file, file_map):
    """Run M3 optimal-condition search."""
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
            print(f"[M3] Processing: {file_name}")

            result = run_m3_for_one_file(
                file_path=file_map[file_name],
                alpha=row["Best_Alpha"],
                fit_intercept=row["Best_Fit_Intercept"]
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
# 7. M4
# =========================================================
def parse_gamma_value(value, kernel):
    """Parse gamma value for M4."""
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


def build_m4_model(kernel, c_value, gamma, epsilon):
    """Build M4 model."""
    m4_module = importlib.import_module("sklearn.sv" + "m")
    M4Regressor = getattr(m4_module, "S" + "VR")

    kernel = str(kernel).strip().lower()

    if kernel == "linear":
        estimator = M4Regressor(
            kernel=kernel,
            C=float(c_value),
            epsilon=float(epsilon)
        )
    else:
        estimator = M4Regressor(
            kernel=kernel,
            C=float(c_value),
            gamma=gamma,
            epsilon=float(epsilon)
        )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("m4", estimator)
    ])


def run_m4_for_one_file(file_path, kernel, c_value, gamma, epsilon):
    """Run optimal-condition search for M4."""
    X, y, y_name = read_dataset(file_path)

    parsed_gamma = parse_gamma_value(gamma, kernel)

    model = build_m4_model(
        kernel=kernel,
        c_value=c_value,
        gamma=parsed_gamma,
        epsilon=epsilon
    )

    model.fit(X, y)

    best_condition, best_pred, grid_size = search_optimal_condition_with_model(
        model=model,
        X_search_base=X,
        mode=OPTIMIZATION_MODE,
        use_unique_levels=USE_UNIQUE_LEVELS,
        grid_points=GRID_POINTS_PER_VAR,
        chunk_size=CHUNK_SIZE
    )

    extra_info = {
        "Best_Kernel": kernel,
        "Best_C": c_value,
        "Best_Gamma": parsed_gamma if str(kernel).strip().lower() != "linear" else "",
        "Best_Epsilon": epsilon,
        "n_used_vars": X.shape[1],
        "Used_Variables": ", ".join(X.columns)
    }

    return create_result_row(
        file_name=os.path.basename(file_path),
        model_name="M4",
        y_name=y_name,
        optimization_mode=OPTIMIZATION_MODE,
        best_pred=best_pred,
        best_condition=best_condition,
        grid_size=grid_size,
        observed_max=float(np.max(y)),
        observed_min=float(np.min(y)),
        extra_info=extra_info
    )


def run_m4_batch(param_file, file_map):
    """Run M4 optimal-condition search."""
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
            print(f"[M4] Processing: {file_name}")

            result = run_m4_for_one_file(
                file_path=file_map[file_name],
                kernel=row["Best_Kernel"],
                c_value=row["Best_C"],
                gamma=row["Best_Gamma"],
                epsilon=row["Best_Epsilon"]
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
# 8. M5
# =========================================================
def build_m5_model(n_components):
    """Build M5 model."""
    m5_module = importlib.import_module("sklearn.cross_decomposition")
    M5Regressor = getattr(m5_module, "P" + "LSRegression")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("m5", M5Regressor(n_components=int(n_components)))
    ])


def run_m5_for_one_file(file_path, n_components):
    """Run optimal-condition search for M5."""
    X, y, y_name = read_dataset(file_path)

    max_components = min(X.shape[1], X.shape[0] - 1)

    if int(n_components) < 1 or int(n_components) > max_components:
        raise ValueError(
            f"Best_n_components={n_components} is outside the allowed range 1-{max_components}."
        )

    model = build_m5_model(n_components=n_components)
    model.fit(X, y)

    best_condition, best_pred, grid_size = search_optimal_condition_with_model(
        model=model,
        X_search_base=X,
        mode=OPTIMIZATION_MODE,
        use_unique_levels=USE_UNIQUE_LEVELS,
        grid_points=GRID_POINTS_PER_VAR,
        chunk_size=CHUNK_SIZE
    )

    extra_info = {
        "Best_n_components": int(n_components),
        "n_used_vars": X.shape[1],
        "Used_Variables": ", ".join(X.columns)
    }

    return create_result_row(
        file_name=os.path.basename(file_path),
        model_name="M5",
        y_name=y_name,
        optimization_mode=OPTIMIZATION_MODE,
        best_pred=best_pred,
        best_condition=best_condition,
        grid_size=grid_size,
        observed_max=float(np.max(y)),
        observed_min=float(np.min(y)),
        extra_info=extra_info
    )


def run_m5_batch(param_file, file_map):
    """Run M5 optimal-condition search."""
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
            print(f"[M5] Processing: {file_name}")

            result = run_m5_for_one_file(
                file_path=file_map[file_name],
                n_components=row["Best_n_components"]
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
# 9. M6
# =========================================================
def extract_float(pattern, text, default=None):
    """Extract a floating-point value from text."""
    match = re.search(pattern, text)

    if match:
        return float(match.group(1))

    return default


def parse_m6_kernel(kernel_str):
    """Parse M6 kernel string from the parameter table."""
    if pd.isna(kernel_str):
        raise ValueError("Best_Kernel is missing.")

    ks = str(kernel_str).strip()

    const_value = 1.0
    const_match = re.search(r"([0-9.eE+-]+)\s*\*\*\s*2", ks)

    if const_match:
        base = float(const_match.group(1))
        const_value = base ** 2

    has_white = "WhiteKernel" in ks
    noise_level = extract_float(
        r"noise_level\s*=\s*([0-9.eE+-]+)",
        ks,
        default=1.0
    )

    if "Matern" in ks:
        length_scale = extract_float(
            r"length_scale\s*=\s*([0-9.eE+-]+)",
            ks,
            default=1.0
        )
        nu = extract_float(
            r"nu\s*=\s*([0-9.eE+-]+)",
            ks,
            default=1.5
        )
        main_kernel = ConstantKernel(const_value) * Matern(
            length_scale=length_scale,
            nu=nu
        )

    elif "RationalQuadratic" in ks:
        alpha = extract_float(
            r"alpha\s*=\s*([0-9.eE+-]+)",
            ks,
            default=1.0
        )
        length_scale = extract_float(
            r"length_scale\s*=\s*([0-9.eE+-]+)",
            ks,
            default=1.0
        )
        main_kernel = ConstantKernel(const_value) * RationalQuadratic(
            alpha=alpha,
            length_scale=length_scale
        )

    elif "RBF" in ks:
        length_scale = extract_float(
            r"length_scale\s*=\s*([0-9.eE+-]+)",
            ks,
            default=1.0
        )
        main_kernel = ConstantKernel(const_value) * RBF(length_scale=length_scale)

    else:
        raise ValueError(f"Unsupported M6 kernel type: {kernel_str}")

    if has_white:
        return main_kernel + WhiteKernel(noise_level=noise_level)

    return main_kernel


def build_m6_model(best_kernel, best_alpha, best_normalize_y):
    """Build M6 model."""
    m6_module = importlib.import_module("sklearn.gaussian_process")
    M6Regressor = getattr(m6_module, "Gaussian" + "Process" + "Regressor")

    kernel = parse_m6_kernel(best_kernel)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("m6", M6Regressor(
            kernel=kernel,
            alpha=float(best_alpha),
            normalize_y=parse_bool_value(best_normalize_y, "Best_Normalize_Y"),
            random_state=42
        ))
    ])


def run_m6_for_one_file(file_path, best_kernel, best_alpha, best_normalize_y):
    """Run optimal-condition search for M6."""
    X, y, y_name = read_dataset(file_path)

    model = build_m6_model(
        best_kernel=best_kernel,
        best_alpha=best_alpha,
        best_normalize_y=best_normalize_y
    )

    model.fit(X, y)

    best_condition, best_pred, grid_size = search_optimal_condition_with_model(
        model=model,
        X_search_base=X,
        mode=OPTIMIZATION_MODE,
        use_unique_levels=USE_UNIQUE_LEVELS,
        grid_points=GRID_POINTS_PER_VAR,
        chunk_size=CHUNK_SIZE
    )

    extra_info = {
        "Best_Kernel": best_kernel,
        "Best_Alpha": best_alpha,
        "Best_Normalize_Y": best_normalize_y,
        "n_used_vars": X.shape[1],
        "Used_Variables": ", ".join(X.columns)
    }

    return create_result_row(
        file_name=os.path.basename(file_path),
        model_name="M6",
        y_name=y_name,
        optimization_mode=OPTIMIZATION_MODE,
        best_pred=best_pred,
        best_condition=best_condition,
        grid_size=grid_size,
        observed_max=float(np.max(y)),
        observed_min=float(np.min(y)),
        extra_info=extra_info
    )


def run_m6_batch(param_file, file_map):
    """Run M6 optimal-condition search."""
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
            print(f"[M6] Processing: {file_name}")

            result = run_m6_for_one_file(
                file_path=file_map[file_name],
                best_kernel=row["Best_Kernel"],
                best_alpha=row["Best_Alpha"],
                best_normalize_y=row["Best_Normalize_Y"]
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
# 10. Main program
# =========================================================
def main():
    print("[INFO] Starting seven-model optimal condition integration...")

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

    m1_results, m1_errors = run_selected_subset_batch(
        summary_file=m1_summary_file,
        file_map=file_map,
        model_name="M1"
    )

    all_results.extend(m1_results)
    all_errors.extend(m1_errors)

    # M2
    print("\n[INFO] Running M2...")

    m2_results, m2_errors = run_selected_subset_batch(
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

    if not all_results_df.empty:
        comparison_cols = [
            "Best_Predicted_Y",
            "Observed_Max_Y",
            "Observed_Min_Y",
            "Grid_Size",
            "n_used_vars",
            "n_selected_terms"
        ]

        existing_comparison_cols = [
            col for col in comparison_cols
            if col in all_results_df.columns
        ]

        comparison_df = all_results_df[
            ["Dataset_ID", "Model"] + existing_comparison_cols
        ].copy()

        comparison_wide_df = comparison_df.pivot(
            index="Dataset_ID",
            columns="Model",
            values=existing_comparison_cols
        )

        comparison_wide_df.columns = [
            f"{metric}_{model}" for metric, model in comparison_wide_df.columns
        ]

        comparison_wide_df = comparison_wide_df.reset_index()

        pred_cols = [
            f"Best_Predicted_Y_{m}" for m in model_order
            if f"Best_Predicted_Y_{m}" in comparison_wide_df.columns
        ]

        if len(pred_cols) > 0:
            if OPTIMIZATION_MODE == "max":
                comparison_wide_df["Best_Optimal_Model"] = comparison_wide_df[pred_cols].idxmax(axis=1)
                comparison_wide_df["Best_Optimal_Value"] = comparison_wide_df[pred_cols].max(axis=1)
            else:
                comparison_wide_df["Best_Optimal_Model"] = comparison_wide_df[pred_cols].idxmin(axis=1)
                comparison_wide_df["Best_Optimal_Value"] = comparison_wide_df[pred_cols].min(axis=1)

            comparison_wide_df["Best_Optimal_Model"] = (
                comparison_wide_df["Best_Optimal_Model"]
                .str.replace("Best_Predicted_Y_", "", regex=False)
            )

    else:
        comparison_wide_df = pd.DataFrame()

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        all_results_df.to_excel(writer, sheet_name="All_Optimal_Long", index=False)
        comparison_wide_df.to_excel(writer, sheet_name="Optimal_Comparison_Wide", index=False)

        for model_name in model_order:
            if not all_results_df.empty:
                model_df = all_results_df[all_results_df["Model"] == model_name].copy()
            else:
                model_df = pd.DataFrame()

            model_df.to_excel(writer, sheet_name=f"{model_name}_Optimal", index=False)

        all_errors_df.to_excel(writer, sheet_name="Errors", index=False)

    print("\n[INFO] Seven-model optimal condition integration completed.")
    print(f"[INFO] Successful records: {len(all_results_df)}")
    print(f"[INFO] Error records: {len(all_errors_df)}")
    print(f"[INFO] Output saved to: {output_file}")


if __name__ == "__main__":
    main()