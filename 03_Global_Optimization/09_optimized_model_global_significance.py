import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import ast

# ==============================================================================
# Script: 09_optimized_model_global_significance.py
# Description: Reconstructs the design matrix for optimized models (M1/M2) 
#              based on selected subsets. Calculates the global F-test p-value 
#              to ensure overall model significance.
# ==============================================================================

def construct_dynamic_matrix(df, terms):
    """
    Dynamically constructs the design matrix from a list of terms.
    Supports: Main effects, Interactions ("A B"), and Quadratics ("A^2").
    """
    X = pd.DataFrame(index=df.index)
    colnames = [col.strip() for col in df.columns]
    df.columns = colnames # Normalize columns

    for term in terms:
        term = term.strip()
        try:
            if term.endswith("^2"):  # Quadratic term
                base = term.replace("^2", "").strip()
                X[term] = df[base] ** 2
            elif ' ' in term:  # Interaction term (e.g., "VarA VarB")
                var1, var2 = term.split(' ', 1)
                X[term] = df[var1.strip()] * df[var2.strip()]
            else:  # Main effect
                X[term] = df[term]
        except Exception as e:
            raise ValueError(f"Failed to construct term '{term}': {e}")
    return X

def calculate_global_p_values(raw_data_dir, best_combo_file, output_file):
    """
    Fits OLS for each dataset using its best-selected combination and
    extracts the global model p-value.
    """
    if not os.path.exists(best_combo_file):
        print(f"[ERROR] Optimization results file not found: {best_combo_file}")
        return

    print(f"[INFO] Initializing Global Model Significance (F-test) Validation...")

    # Load the optimization results (M1 or M2)
    combo_df = pd.read_excel(best_combo_file)
    
    # Identify key columns (supporting both Chinese and English headers)
    id_col = next((c for c in combo_df.columns if 'Dataset' in str(c) or '数据集' in str(c)), None)
    combo_col = next((c for c in combo_df.columns if 'Combination' in str(c) or '最佳组合' in str(c)), None)

    if not id_col or not combo_col:
        print(f"[ERROR] Could not identify ID or Combination columns in: {combo_df.columns.tolist()}")
        return

    results = []

    for idx, row in combo_df.iterrows():
        dataset_name = row[id_col]
        combo_str = str(row[combo_col])

        # Parse the combination string (handles lists or CSV strings)
        try:
            if combo_str.startswith('[') and combo_str.endswith(']'):
                var_list = ast.literal_eval(combo_str)
            else:
                var_list = [item.strip() for item in combo_str.split(',') if item.strip()]
        except:
            results.append({"Dataset_ID": dataset_name, "Global_P_Value": "Parse Error"})
            continue

        data_path = os.path.join(raw_data_dir, dataset_name)
        if not os.path.exists(data_path):
            results.append({"Dataset_ID": dataset_name, "Global_P_Value": "File Missing"})
            continue

        try:
            # Load raw data (Reads primary worksheet)
            df_raw = pd.read_excel(data_path)
            y = df_raw.iloc[:, -1].astype(float)
            X_orig = df_raw.iloc[:, :-1]

            # Reconstruct model-specific design matrix
            X_custom = construct_dynamic_matrix(X_orig, var_list)
            X_with_const = sm.add_constant(X_custom, has_constant='add')

            # Degrees of freedom check (n > p)
            if X_with_const.shape[0] <= X_with_const.shape[1]:
                f_pvalue = "Insufficient DF"
            else:
                # Fit OLS and extract global F-test p-value
                model = sm.OLS(y, X_with_const).fit()
                f_pvalue = round(model.f_pvalue, 6)

            results.append({
                "Dataset_ID": dataset_name, 
                "Selected_Terms": ", ".join(var_list), 
                "Global_P_Value": f_pvalue
            })
            print(f"[ANALYSIS] {dataset_name} | Global P: {f_pvalue}")

        except Exception as e:
            results.append({"Dataset_ID": dataset_name, "Global_P_Value": f"Error: {str(e)}"})

    # Save summary report
    if results:
        output_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output_df.to_excel(output_file, index=False)
        print("-" * 60)
        print(f"[COMPLETE] Global P-value validation finished.")
        print(f"[INFO] Report generated at: {output_file}")
    else:
        print("[WARN] No results generated.")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Replace with directory containing your raw experimental data
    RAW_DATA_DIRECTORY = r"YOUR_RAW_DATA_PATH_HERE"
    
    # [TODO] Replace with the path to your M1 or M2 optimization results (Excel)
    OPTIMIZATION_RESULTS_XLS = r"YOUR_OPTIMIZATION_RESULTS_PATH_HERE.xlsx"
    
    # [TODO] Desired output path for the global p-value report
    GLOBAL_P_VALUE_REPORT = r"YOUR_OUTPUT_REPORT_PATH_HERE\Global_Significance_Report.xlsx"

    # -------------------------------------------------------------------------
    calculate_global_p_values(RAW_DATA_DIRECTORY, OPTIMIZATION_RESULTS_XLS, GLOBAL_P_VALUE_REPORT)