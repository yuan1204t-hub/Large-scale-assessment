import os
import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon

# ==============================================================================
# Script: 08_model_performance_comparison_test.py
# Description: Conducts paired statistical tests to compare Adjusted R2 
#              between two modeling strategies (e.g., M0 vs M1).
#              Determines if the performance gain is statistically significant.
# ==============================================================================

def compare_model_performance(file_ref, file_opt, col_ref="Adjusted_R2", col_opt="Adj_R2"):
    """
    Performs paired analysis:
    1. Shapiro-Wilk test on differences to check normality.
    2. Paired t-test (parametric) or Wilcoxon signed-rank test (non-parametric).
    """
    if not os.path.exists(file_ref) or not os.path.exists(file_opt):
        print(f"[ERROR] Input files not found. Please check your paths.")
        return

    print(f"[INFO] Initializing Significance Test for Model Comparison...")
    print(f"[INFO] Reference Model: {os.path.basename(file_ref)}")
    print(f"[INFO] Optimized Model: {os.path.basename(file_opt)}")

    try:
        # Load datasets (Defaults to the first worksheet)
        df_ref = pd.read_excel(file_ref)
        df_opt = pd.read_excel(file_opt)

        # Extract Adjusted R2 values
        # Ensuring numeric conversion and handling missing data
        x = pd.to_numeric(df_ref[col_ref], errors="coerce").dropna().values
        y = pd.to_numeric(df_opt[col_opt], errors="coerce").dropna().values

        # Align lengths (Ensure we are comparing the same datasets)
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]
        
        # Calculate differences for the normality test
        diff = x - y

        # --- 1. Normality Test of Differences (Shapiro-Wilk) ---
        sh_stat, p_norm = shapiro(diff)
        is_normal = p_norm > 0.05
        
        # --- 2. Paired Comparison Test Selection ---
        if is_normal:
            test_method = "Paired t-test (Parametric)"
            # Tests if the mean of differences is significantly different from 0
            t_stat, p_val = ttest_rel(x, y)
        else:
            test_method = "Wilcoxon Signed-Rank Test (Non-parametric)"
            # Tests if the median of differences is significantly different from 0
            try:
                t_stat, p_val = wilcoxon(x, y)
            except ValueError as e:
                t_stat, p_val = np.nan, np.nan
                print(f"[WARN] Wilcoxon test failed: {e}")

        # --- 3. Output Statistical Summary ---
        print("\n" + "="*60)
        print("📊 STATISTICAL COMPARISON SUMMARY")
        print("="*60)
        print(f"Comparison Group  : {os.path.basename(file_ref)} vs {os.path.basename(file_opt)}")
        print(f"Sample Size (n)   : {min_len}")
        print(f"Normality (p-val) : {p_norm:.4e} ({'Normal' if is_normal else 'Non-normal'})")
        print(f"Applied Method    : {test_method}")
        print(f"Statistical p-val : {p_val:.4e}")
        print("-" * 60)
        
        if p_val is not None and p_val < 0.05:
            print(f"✨ RESULT: The performance difference is **SIGNIFICANT** ($p < 0.05$).")
            print(f"   Mean $R^2_{{adj}}$ improved from {x.mean():.4f} to {y.mean():.4f}")
        else:
            print(f"⚠️ RESULT: The performance difference is **NOT SIGNIFICANT** ($p \geq 0.05$).")
        print("="*60)

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during analysis: {e}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Replace with your absolute path to the reference results (e.g., M0)
    REFERENCE_RESULTS = r"YOUR_PATH_TO_M0_RESULTS.xlsx"
    
    # [TODO] Replace with your absolute path to the optimized results (e.g., M1 or M2)
    OPTIMIZED_RESULTS = r"YOUR_PATH_TO_M1_RESULTS.xlsx"
    
    # [TODO] Ensure the column names match your Excel headers exactly

    REF_COL_NAME = "R2" 
    OPT_COL_NAME = "Avg_Adjusted_R2"

    # -------------------------------------------------------------------------
    compare_model_performance(REFERENCE_RESULTS, OPTIMIZED_RESULTS, 
                              col_ref=REF_COL_NAME, col_opt=OPT_COL_NAME)