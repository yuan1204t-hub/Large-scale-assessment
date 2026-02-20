import pandas as pd
import numpy as np
from scipy.stats import shapiro
import os

# ==============================================================================
# Script: 12_optimized_pmax_normality_test.py
# Section: 3.2 Optimal Modeling Strategy
# Description: Conducts the Shapiro-Wilk test to evaluate the normality of the 
#              Maximum p-value (p_max) distribution in optimized models. 
#              The result determines the appropriate post-hoc statistical 
#              methodology (Parametric vs. Non-parametric).
# ==============================================================================

def analyze_pmax_distribution(file_path):
    """
    Loads optimization results and performs a statistical normality test 
    on the significance levels (max p-values).
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] Input file not found: {file_path}")
        return

    print(f"[INFO] Starting Normality Assessment for Optimized Model p_max...")

    try:
        # 1. Load Data (Reads the first worksheet by default)
        df = pd.read_excel(file_path)
        
        # 2. Extract 'Maximum p-value' column
        # Mapping logic: searching for common header patterns
        col_name = next((c for c in df.columns if 'max' in str(c).lower() or '最大' in str(c)), None)
        
        if not col_name:
            print(f"[ERROR] Required column (p_max) not found in headers: {df.columns.tolist()}")
            return
            
        # Standardize numeric conversion
        data = pd.to_numeric(df[col_name], errors='coerce').dropna()

        # 3. Descriptive Statistics
        print(f"\n[ANALYSIS] Descriptive Statistics for '{col_name}':")
        print(f"  - Sample Size (n): {len(data)}")
        print(f"  - Mean Value     : {data.mean():.6f}")
        print(f"  - Median Value   : {data.median():.6f}")
        print(f"  - Std. Deviation : {data.std():.6f}")
        print("-" * 60)

        # 4. Shapiro-Wilk Normality Test
        # H0: The data is normally distributed.
        stat, p_value = shapiro(data)

        print(f"[STATUS] Shapiro-Wilk Test Results:")
        print(f"  - Statistic (W)  : {stat:.6f}")
        print(f"  - p-value        : {p_value:.6e}")
        print("-" * 60)

        # 5. Statistical Conclusion Logic (Logic Preserved)
        if p_value > 0.05:
            print("[RESULT] Conclusion: The distribution follows a NORMAL profile ($p > 0.05$).")
            print("[INFO] Recommendation: Use Parametric tests (e.g., Paired t-test) for subsequent comparisons.")
        else:
            print("[RESULT] Conclusion: The distribution is NON-NORMAL ($p \leq 0.05$).")
            print("[INFO] Recommendation: Use Non-parametric tests (e.g., Wilcoxon Signed-Rank test).")

    except Exception as e:
        print(f"[ERROR] Statistical analysis failed: {e}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    
    # [TODO] Replace with your absolute path to the optimization result file
    INPUT_DATA_PATH = r"YOUR_OPTIMIZATION_RESULT_FILE_HERE.xlsx"
    
    # -------------------------------------------------------------------------
    analyze_pmax_distribution(INPUT_DATA_PATH)