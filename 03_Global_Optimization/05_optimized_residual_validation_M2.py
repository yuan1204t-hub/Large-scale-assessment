import os
import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_1samp, wilcoxon

# ==============================================================================
# Script: 05_optimized_residual_validation_M2.py
# Description: Validates the residuals of Model 2 (M2, Cp-based). 
#              Handles potential string-formatted prediction lists and evaluates 
#              normality (p_n) and unbiasedness (p_m).
# ==============================================================================

def validate_m2_residuals(input_file, output_file):
    """
    Analyzes residuals for Model 2 datasets. 
    Compares Actual vs. Predicted values to ensure statistical validity.
    """
    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        return

    print(f"[INFO] Initializing residual diagnostics for Model 2 (Cp-criterion): {os.path.basename(input_file)}")

    try:
        # Load optimized results
        df = pd.read_excel(input_file)
        results = []
        
        # Summary counters
        normality_pass = 0
        bias_pass = 0
        total_count = 0

        # Group processing for each dataset (Standardized Column: Dataset_ID)
        # Assuming M2 optimization output uses 'Dataset_ID'
        group_col = "Dataset_ID" 

        for name, group in df.groupby(group_col):
            total_count += 1
            all_pred = []
            all_true = []

            # Robust parsing for potential CSV-like strings in cells (e.g., "0.5, 0.6")
            for _, row in group.iterrows():
                try:
                    # Clean and parse Predicted vs Actual strings
                    y_p = [float(x.strip()) for x in str(row["Predicted"]).split(',') if x.strip()]
                    y_t = [float(x.strip()) for x in str(row["Actual"]).split(',') if x.strip()]

                    if len(y_p) == len(y_t):
                        all_pred.extend(y_p)
                        all_true.extend(y_t)
                except ValueError:
                    continue

            all_pred = np.array(all_pred)
            all_true = np.array(all_true)

            if len(all_pred) < 3:
                print(f"[SKIP] {name}: Sample size insufficient for statistical testing.")
                continue

            # Residual calculation: e = Actual - Predicted
            residuals = all_true - all_pred

            # 1. Normality Test (Shapiro-Wilk)
            sh_stat, p_n = shapiro(residuals)
            is_normal = p_n > 0.05
            if is_normal: normality_pass += 1

            # 2. Unbiasedness Test (Mean = 0)
            # Parametric t-test if normal, otherwise non-parametric Wilcoxon
            if is_normal:
                test_used = "t-test"
                _, p_m = ttest_1samp(residuals, 0)
            else:
                test_used = "Wilcoxon"
                try:
                    _, p_m = wilcoxon(residuals)
                except Exception:
                    p_m = np.nan

            is_unbiased = p_m > 0.05 if not np.isnan(p_m) else False
            if is_unbiased: bias_pass += 1

            results.append({
                "Dataset_ID": name,
                "Shapiro_Stat": round(sh_stat, 4),
                "Normality_p": round(p_n, 4),
                "Is_Normal": "Yes" if is_normal else "No",
                "Test_Method": test_used,
                "Bias_p_value": round(p_m, 4) if not np.isnan(p_m) else "N/A"
            })
            print(f"[STATUS] Diagnosed: {name}")

        # Save findings
        if results:
            final_df = pd.DataFrame(results)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            final_df.to_excel(output_file, index=False)

            # Print professional summary for the manuscript
            print("\n" + "="*60)
            print(f"[ANALYSIS] M2 (Cp-criterion) Residual Evaluation Summary (n={total_count})")
            print("-" * 60)
            print(f"Normality Adherence (p_n > 0.05): {normality_pass} ({ (normality_pass/total_count)*100:.2f}%)")
            print(f"Zero-Bias Adherence (p_m > 0.05): {bias_pass} ({ (bias_pass/total_count)*100:.2f}%)")
            print("-" * 60)
            print(f"[INFO] Diagnostic report saved to: {output_file}")
            print("="*60)
        else:
            print("[WARN] No valid data groups found for analysis.")

    except Exception as e:
        print(f"[ERROR] An unexpected system failure occurred: {e}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Replace with your M2 optimization result path (containing predictions)
    INPUT_FILE_PATH = r"YOUR_M2_PREDICTION_RESULTS_XLSX"
    
    # [TODO] Replace with your desired output path for the M2 residual report
    M2_RESIDUAL_REPORT = r"YOUR_OUTPUT_REPORT_PATH_HERE\M2_Residual_Validation.xlsx"
    
    # -------------------------------------------------------------------------
    validate_m2_residuals(INPUT_FILE_PATH, M2_RESIDUAL_REPORT)