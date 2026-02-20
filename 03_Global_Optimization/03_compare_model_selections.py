import os
import re
import pandas as pd
import ast

# ==============================================================================
# Script: 03_variable_selection_comparison.py
# Description: Compares selected variable sets from two different optimization 
#              models (e.g., M1 vs M2). Standardizes list-like and CSV-like 
#              string formats into sets for unordered equality verification.
# ==============================================================================

def normalize_format_list(x):
    """
    Parses the first format type: string representations of lists 
    e.g., "['A', 'B', 'C', 'A^2']"
    """
    if pd.isna(x) or not str(x).strip():
        return set()
    try:
        # Use ast.literal_eval for safe string-to-list conversion
        items = ast.literal_eval(x) if isinstance(x, str) else x
        return set(str(i).strip().replace('"', '').replace("'", '') for i in items)
    except Exception:
        return set()

def normalize_format_csv(x):
    """
    Parses the second format type: comma-separated strings
    e.g., "A, B, C, A^2, A B"
    """
    if pd.isna(x) or not str(x).strip():
        return set()
    items = str(x).split(',')
    # Exclude intercept '1' and clean whitespace
    return set(i.strip() for i in items if i.strip() and i.strip() != '1')

def compare_variable_selections(file1_path, file2_path, output_path):
    """
    Main comparison logic: Normalizes columns and identifies identity vs divergence.
    """
    # Standard column names as per previous script outputs
    dataset_col = 'Dataset_ID' 
    factor_col = 'Best_Combination'

    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        print(f"[ERROR] One or both input files not found.")
        return

    try:
        print("[INFO] Loading model optimization results for comparison...")
        # Load datasets (using standard column headers)
        df1 = pd.read_excel(file1_path)
        df2 = pd.read_excel(file2_path)

        # Apply normalization logic based on expected input format
        df1['norm_vars'] = df1[factor_col].apply(normalize_format_list)
        df2['norm_vars'] = df2[factor_col].apply(normalize_format_csv)

        # Merge on Dataset_ID to ensure row-wise alignment
        merged = pd.merge(df1, df2, on=dataset_col, suffixes=('_M1', '_M2'))
        
        results_list = []
        identical_count = 0
        different_count = 0

        print("[INFO] Executing set-based combination comparison...")
        
        for _, row in merged.iterrows():
            # Set-based comparison ignores order (e.g., {A, B} == {B, A})
            if row['norm_vars_M1'] == row['norm_vars_M2']:
                results_list.append({
                    'Identical_Combination_ID': row[dataset_col], 
                    'Divergent_Combination_ID': None,
                    'Selection': ", ".join(sorted(list(row['norm_vars_M1'])))
                })
                identical_count += 1
            else:
                results_list.append({
                    'Identical_Combination_ID': None, 
                    'Divergent_Combination_ID': row[dataset_col],
                    'M1_Selection': ", ".join(sorted(list(row['norm_vars_M1']))),
                    'M2_Selection': ", ".join(sorted(list(row['norm_vars_M2'])))
                })
                different_count += 1

        # Create results dataframe
        result_df = pd.DataFrame(results_list)

        # Export findings
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_excel(output_path, index=False)
        
        # Performance Statistics
        total = identical_count + different_count
        identical_pct = (identical_count / total) * 100 if total > 0 else 0
        
        print("-" * 60)
        print(f"[SUMMARY] Variable Selection Consistency Report")
        print(f"Total Comparison   : {total} datasets")
        print(f"Identical Sets (✔) : {identical_count} ({identical_pct:.2f}%)")
        print(f"Divergent Sets (✘) : {different_count} ({(100-identical_pct):.2f}%)")
        print(f"[STATUS] Detailed report generated: {output_path}")
        print("-" * 60)

    except Exception as e:
        print(f"[ERROR] An exception occurred during comparison: {e}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Path to the first model results (e.g., M1 optimized results)
    MODEL_1_RESULTS = r"YOUR_MODEL_1_PATH_HERE.xlsx"
    
    # [TODO] Path to the second model results (e.g., M2 optimized results)
    MODEL_2_RESULTS = r"YOUR_MODEL_2_PATH_HERE.xlsx"
    
    # [TODO] Final output path for the comparison report
    COMPARISON_REPORT = r"YOUR_OUTPUT_PATH_HERE\Variable_Comparison_Report.xlsx"
    
    # -------------------------------------------------------------------------
    compare_variable_selections(MODEL_1_RESULTS, MODEL_2_RESULTS, COMPARISON_REPORT)