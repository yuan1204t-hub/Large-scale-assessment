import os
import pandas as pd
from scipy.stats import ttest_rel

# ==============================================================================
# Script: 13_paired_t_test_r2_comparison.py
# Description: Performs paired t-tests on R2 values across MATLAB, Python, and R.
#              Since the same datasets are used across platforms, paired testing
#              provides higher statistical power to detect platform-induced 
#              discrepancies.
# ==============================================================================

def perform_paired_comparison(file_path, output_path):
    """
    Reads the software summary file and executes paired t-tests for each 
    platform pair to evaluate the consistency of model fitting performance.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] Summary file not found: {file_path}")
        return

    print(f"[INFO] Initializing paired t-test analysis: {os.path.basename(file_path)}")

    try:
        # Load the software aggregation summary
        df = pd.read_excel(file_path)
        
        # Clean data: drop missing values for specific columns
        # Aligning with standardized column names from Script 11
        cols = ['MATLAB_R2', 'Python_R2', 'R_R2']
        if not all(c in df.columns for c in cols):
            print(f"[ERROR] Missing required columns in dataset. Expected: {cols}")
            return

        clean_df = df[cols].dropna()
        
        matlab_r2 = clean_df['MATLAB_R2']
        python_r2 = clean_df['Python_R2']
        r_r2 = clean_df['R_R2']

        # Initialize results structure
        results = {
            'Comparison_Pair': ['MATLAB vs Python', 'MATLAB vs R', 'Python vs R'],
            't_statistic': [],
            'p_value': [],
            'Significant_at_0.05': []
        }

        print("[INFO] Computing paired differences across platform pairs...")

        # 1. MATLAB vs Python
        t_stat1, p_val1 = ttest_rel(matlab_r2, python_r2)
        results['t_statistic'].append(t_stat1)
        results['p_value'].append(p_val1)
        results['Significant_at_0.05'].append('Yes' if p_val1 < 0.05 else 'No')

        # 2. MATLAB vs R
        t_stat2, p_val2 = ttest_rel(matlab_r2, r_r2)
        results['t_statistic'].append(t_stat2)
        results['p_value'].append(p_val2)
        results['Significant_at_0.05'].append('Yes' if p_val2 < 0.05 else 'No')

        # 3. Python vs R
        t_stat3, p_val3 = ttest_rel(python_r2, r_r2)
        results['t_statistic'].append(t_stat3)
        results['p_value'].append(p_val3)
        results['Significant_at_0.05'].append('Yes' if p_val3 < 0.05 else 'No')

        # Save to Excel
        result_df = pd.DataFrame(results)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        result_df.to_excel(output_path, index=False)

        # Print summary for immediate verification
        print("\n" + "="*60)
        print(f"[ANALYSIS] Paired t-test Statistical Summary (n={len(clean_df)})")
        print("-" * 60)
        for i, pair in enumerate(results['Comparison_Pair']):
            p_val = results['p_value'][i]
            sig = results['Significant_at_0.05'][i]
            t_val = results['t_statistic'][i]
            print(f"{pair:18}: t = {t_val:8.4f}, p = {p_val:.4e} [Sig: {sig}]")
        print("-" * 60)
        print(f"[STATUS] Detailed results exported to: {output_path}")
        print("="*60)

    except Exception as e:
        print(f"[ERROR] An exception occurred during paired t-test execution: {e}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Replace with the path to your aggregated summary Excel file
    INPUT_SUMMARY_FILE = r"YOUR_SUMMARY_FILE_PATH_HERE"
    
    # [TODO] Replace with the desired output path for paired t-test results
    T_TEST_OUTPUT_PATH = r"YOUR_OUTPUT_DIRECTORY_PATH_HERE\paired_t_test_results.xlsx"
    
    # -------------------------------------------------------------------------
    perform_paired_comparison(INPUT_SUMMARY_FILE, T_TEST_OUTPUT_PATH)