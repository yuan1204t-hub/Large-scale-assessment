# -*- coding: utf-8 -*-
"""
Purpose:
1. Batch calculate pF and pmax for three traditional regression models:
   - M0
   - M1
   - M2

2. For M0:
   A full quadratic OLS model is fitted.

3. For M1 and M2:
   Best_Variables are read from their corresponding summary files,
   and selected-term OLS models are refitted.

4. Output:
   - All_pF_pmax_Long
   - pF_pmax_Comparison_Wide
   - M0_pF_pmax
   - M1_pF_pmax
   - M2_pF_pmax
   - Errors

Definitions:
- pF: overall F-test p-value of the OLS model.
- pmax: maximum coefficient p-value excluding the intercept.
"""

import os
import glob
import re
import ast
import string
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures


# =========================================================
# 1. Path settings
# =========================================================
data_dir = r"Please enter your path here"

m1_summary_file = r"Please enter your path here"
m2_summary_file = r"Please enter your path here"

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
    y_name = df.columns[-1]

    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if len(y) < 3:
        raise ValueError("The number of valid samples is too small.")

    return X, y, y_name


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


def fit_ols_and_calculate_pf_pmax(X_model, y):
    """
    Fit an OLS model and calculate pF and pmax.

    pF:
    Overall F-test p-value.

    pmax:
    Maximum coefficient p-value excluding the intercept.
    """
    X_const = sm.add_constant(X_model, has_constant="add")

    if len(X_const) <= X_const.shape[1]:
        raise ValueError(
            f"Insufficient sample size for OLS fitting: "
            f"n={len(X_const)}, parameters={X_const.shape[1]}"
        )

    model = sm.OLS(y, X_const).fit()

    pvalues_no_const = model.pvalues.drop("const", errors="ignore")

    if len(pvalues_no_const) == 0:
        pmax = np.nan
    else:
        pmax = float(np.nanmax(pvalues_no_const.values))

    return {
        "pF": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        "pmax": pmax,
        "R2_refit": float(model.rsquared),
        "Adjusted_R2_refit": float(model.rsquared_adj),
        "AIC_refit": float(model.aic),
        "BIC_refit": float(model.bic),
        "Nt": X_const.shape[1] - 1,
        "n_used": len(y)
    }


# =========================================================
# 3. M0: full quadratic OLS model
# =========================================================
def build_m0_design_matrix(X_raw):
    """Build full quadratic design matrix for M0."""
    X_full = pd.DataFrame(index=X_raw.index)
    cols = list(X_raw.columns)

    # Linear terms
    for col in cols:
        X_full[col] = X_raw[col]

    # Squared terms
    for col in cols:
        X_full[f"{col}^2"] = X_raw[col] ** 2

    # Two-way interaction terms
    for c1, c2 in itertools.combinations(cols, 2):
        X_full[f"{c1} {c2}"] = X_raw[c1] * X_raw[c2]

    return X_full


def run_m0_for_one_file(file_path):
    """Calculate pF and pmax for M0."""
    X_raw, y, y_name = read_dataset(file_path)

    X_model = build_m0_design_matrix(X_raw)
    metrics = fit_ols_and_calculate_pf_pmax(X_model, y)

    return {
        "Dataset_ID": os.path.basename(file_path),
        "Model": "M0",
        "Y_Name": y_name,
        "Best_Variables": "Full quadratic model",
        "Used_Variables": ", ".join(X_raw.columns),
        **metrics
    }


# =========================================================
# 4. M1 and M2: selected-term OLS models
# =========================================================
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

    # Try original variable names first
    used_vars = [v for v in raw_vars if v in base_candidates]

    if used_vars:
        return used_vars, False

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

    return used_vars, True


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


def run_selected_subset_for_one_file(file_path, best_variables_text, model_name):
    """Calculate pF and pmax for M1 or M2."""
    X_all, y, y_name = read_dataset(file_path)

    raw_vars = list(X_all.columns)
    best_terms = parse_combo(best_variables_text)

    if len(best_terms) == 0:
        raise ValueError("Best_Variables is empty.")

    used_vars, used_coded_names = get_used_vars_from_terms(best_terms, raw_vars)

    X_used = X_all[used_vars].copy()

    if used_coded_names:
        letters = list(string.ascii_uppercase)
        X_used_for_poly = X_used.copy()
        X_used_for_poly.columns = letters[:len(used_vars)]
        poly_feature_base_names = list(X_used_for_poly.columns)
    else:
        X_used_for_poly = X_used.copy()
        poly_feature_base_names = used_vars

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(X_used_for_poly)

    all_features = poly.get_feature_names_out(poly_feature_base_names)
    selected_idx = match_selected_polynomial_terms(all_features, best_terms)

    X_poly = poly.transform(X_used_for_poly)[:, selected_idx]
    selected_feature_names = [all_features[i] for i in selected_idx]

    X_model = pd.DataFrame(
        X_poly,
        columns=selected_feature_names,
        index=X_used.index
    )

    metrics = fit_ols_and_calculate_pf_pmax(X_model, y)

    return {
        "Dataset_ID": os.path.basename(file_path),
        "Model": model_name,
        "Y_Name": y_name,
        "Best_Variables": ", ".join(best_terms),
        "Used_Variables": ", ".join(used_vars),
        **metrics
    }


def run_selected_subset_batch(summary_file, file_map, model_name):
    """Run pF and pmax calculation for M1 or M2."""
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

            result = run_selected_subset_for_one_file(
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
# 5. Main program
# =========================================================
def main():
    print("[INFO] Starting M0-M2 pF/pmax integration...")

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

            result = run_m0_for_one_file(file_path)
            all_results.append(result)

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

    m1_results, m1_errors = run_selected_subset_batch(
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

    m2_results, m2_errors = run_selected_subset_batch(
        summary_file=m2_summary_file,
        file_map=file_map,
        model_name="M2"
    )

    all_results.extend(m2_results)
    all_errors.extend(m2_errors)

    # =====================================================
    # Save outputs
    # =====================================================
    all_results_df = pd.DataFrame(all_results)
    all_errors_df = pd.DataFrame(all_errors)

    model_order = ["M0", "M1", "M2"]

    if not all_results_df.empty:
        all_results_df["Model"] = pd.Categorical(
            all_results_df["Model"],
            categories=model_order,
            ordered=True
        )

        all_results_df = all_results_df.sort_values(["Dataset_ID", "Model"])

    comparison_metrics = [
        "pF",
        "pmax",
        "R2_refit",
        "Adjusted_R2_refit",
        "AIC_refit",
        "BIC_refit",
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

        # Pass indicators
        for model_name in model_order:
            pf_col = f"pF_{model_name}"
            pmax_col = f"pmax_{model_name}"

            if pf_col in comparison_wide_df.columns:
                comparison_wide_df[f"pF_Pass_0.05_{model_name}"] = (
                    comparison_wide_df[pf_col] < 0.05
                )

            if pmax_col in comparison_wide_df.columns:
                comparison_wide_df[f"pmax_Pass_0.05_{model_name}"] = (
                    comparison_wide_df[pmax_col] < 0.05
                )

    else:
        comparison_wide_df = pd.DataFrame()

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        all_results_df.to_excel(writer, sheet_name="All_pF_pmax_Long", index=False)
        comparison_wide_df.to_excel(writer, sheet_name="pF_pmax_Comparison_Wide", index=False)

        for model_name in model_order:
            if not all_results_df.empty:
                model_df = all_results_df[all_results_df["Model"] == model_name].copy()
            else:
                model_df = pd.DataFrame()

            model_df.to_excel(writer, sheet_name=f"{model_name}_pF_pmax", index=False)

        all_errors_df.to_excel(writer, sheet_name="Errors", index=False)

    print("\n[INFO] M0-M2 pF/pmax integration completed.")
    print(f"[INFO] Successful records: {len(all_results_df)}")
    print(f"[INFO] Error records: {len(all_errors_df)}")
    print(f"[INFO] Output saved to: {output_file}")


if __name__ == "__main__":
    main()