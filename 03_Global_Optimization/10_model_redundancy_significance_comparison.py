import os
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

# ==============================================================================
# Script: 09_model_redundancy_significance_comparison.py
# Description: Performs Friedman test and post-hoc Wilcoxon tests (with Bonferroni)
#              on the maximum p-values (p_max) of different modeling strategies.
#              Provides statistical evidence for model redundancy reduction.
# ==============================================================================

def analyze_model_redundancy_significance(file_path, output_report_path):
    """
    Evaluates whether there is a significant difference in the significance level 
    (p_max) between Full (M0), M1, and M2 models.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] Input summary file not found: {file_path}")
        return

    print(f"[INFO] Initializing Non-parametric Redundancy Analysis...")

    try:
        # 1. Load data (Defaults to the first worksheet)
        df = pd.read_excel(file_path)
        
        # Identify p_max columns for M0, M1, and M2
        # We look for columns containing p-value data for the three strategies
        cols = [c for c in df.columns if 'p' in str(c).lower() or '最大' in str(c)]
        
        if len(cols) < 3:
            print(f"[ERROR] Found only {len(cols)} p-value columns. Need at least 3 (M0, M1, M2).")
            print(f"[INFO] Available columns: {df.columns.tolist()}")
            return

        data = df[cols[:3]].dropna()
        n_samples = len(data)
        
        # Mapping for clarity in reports
        data.columns = ['Max_P_Full', 'Max_P_M1', 'Max_P_M2']

        print(f"[STATUS] Successfully loaded {n_samples} datasets for comparison.")

        # 2. Global Test: Friedman Test
        # Ideal for non-normal p-value distributions across related groups
        stat_f, p_f = friedmanchisquare(data['Max_P_Full'], data['Max_P_M1'], data['Max_P_M2'])
        
        print("\n" + "="*60)
        print(f"📊 GLOBAL ANALYSIS: Friedman Test Result")
        print(f"------------------------------------------------------------")
        print(f"Statistic (Q) : {stat_f:.4e}")
        print(f"P-value       : {p_f:.4e}")
        print("="*60)

        if p_f >= 0.05:
            print(f"[RESULT] No significant difference in $p_{{max}}$ across strategies ($p \geq 0.05$).")
            print("[INFO] Post-hoc analysis bypassed.")
            return

        print(f"[RESULT] Significant differences detected ($p < 0.05$).")
        print("[INFO] Proceeding to Pairwise Post-hoc comparisons (Wilcoxon + Bonferroni Correction)...")

        # 3. Post-hoc: Pairwise Wilcoxon Signed-Rank Tests
        pairs = [
            ('Full vs M1', data['Max_P_Full'], data['Max_P_M1']),
            ('Full vs M2', data['Max_P_Full'], data['Max_P_M2']),
            ('M1 vs M2', data['Max_P_M1'], data['Max_P_M2'])
        ]
        
        raw_p_values = []
        for label, group_a, group_b in pairs:
            _, p_w = wilcoxon(group_a, group_b)
            raw_p_values.append(p_w)

        # 4. Multiple Comparison Correction (Bonferroni)
        # Crucial to control Type I error inflation in multiple hypothesis testing
        reject, p_corrected, _, _ = multipletests(raw_p_values, method='bonferroni')

        # 5. Result Aggregation and Reporting
        comparison_results = []
        print("\n" + "-"*60)
        print(f"{'Comparison Pair':<20} | {'Raw P-val':<12} | {'Adj P-val':<12} | {'Result'}")
        print("-" * 60)
        
        for i, (label, _, _) in enumerate(pairs):
            sig_status = "SIGNIFICANT" if reject[i] else "Not Significant"
            print(f"{label:<20} | {raw_p_values[i]:.4e} | {p_corrected[i]:.4e} | {sig_status}")
            comparison_results.append({
                "Comparison_Pair": label,
                "Raw_P_Value": raw_p_values[i],
                "Adjusted_P_Value_Bonferroni": p_corrected[i],
                "Is_Significant": "Yes" if reject[i] else "No"
            })

        # Export Excel Report
        os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
        pd.DataFrame(comparison_results).to_excel(output_report_path, index=False)
        
        print("-" * 60)
        print(f"[COMPLETE] Statistical redundancy report saved to:")
        print(f" -> {output_report_path}")
        print("="*60)

    except Exception as e:
        print(f"[ERROR] An exception occurred during statistical analysis: {e}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Replace with your summarized max_p values Excel file path
    INPUT_MAX_P_XLS = r"YOUR_PATH_TO_MAX_P_VALUES.xlsx"
    
    # [TODO] Replace with your desired output path for the redundancy report
    OUTPUT_REPORT_PATH = r"YOUR_OUTPUT_REPORT_PATH_HERE\Redundancy_Statistical_Analysis.xlsx"
    
    # -------------------------------------------------------------------------
    analyze_model_redundancy_significance(INPUT_MAX_P_XLS, OUTPUT_REPORT_PATH)