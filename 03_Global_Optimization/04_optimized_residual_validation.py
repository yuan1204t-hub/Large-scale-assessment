import os
import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_1samp, wilcoxon

# ==============================================================================
# Script: 04_optimized_residual_validation.py
# Description: Evaluates the residual validity of the optimized model (M1).
#              Performs Shapiro-Wilk test for normality and Mean-Zero test 
#              (t-test or Wilcoxon) to ensure the optimized model is unbiased.
# ==============================================================================

def validate_optimized_residuals(input_file, output_file):
    """
    Analyzes the prediction residuals from the LOOCV process of Model 1.
    Calculates residual normality (p_n) and mean bias (p_m).
    """
    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        return

    print(f"[INFO] Initializing residual diagnostics for Model 1: {os.path.basename(input_file)}")

    try:
        # Load the LOOCV results containing 'Predicted' and 'Actual' values
        df = pd.read_excel(input_file)
        
        # Verify required columns exist
        required_cols = ['Dataset_ID', 'Predicted', 'Actual']
        if not all(col in df.columns for col in required_cols):
            print(f"[ERROR] Missing required columns. Expected: {required_cols}")
            return

        results = []
        passed_normality = 0
        passed_bias_test = 0
        total_groups = 0

        # Group by dataset to evaluate residuals per model
        grouped = df.groupby('Dataset_ID')
        print(f"[INFO] Analyzing {len(grouped)} distinct dataset models...")

        for name, group in grouped:
            total_groups += 1
            y_pred = group['Predicted'].values
            y_true = group['Actual'].values
            
            # Residual calculation
            residuals = y_pred - y_true

            # 1. Normality Test (Shapiro-Wilk)
            sh_stat, p_n = shapiro(residuals)
            is_normal = p_n > 0.05
            if is_normal: passed_normality += 1

            # 2. Mean-Zero Test (Unbiasedness)
            # If normal, use t-test; otherwise, use Wilcoxon signed-rank test
            if is_normal:
                method = 't-test'
                _, p_m = ttest_1samp(residuals, popmean=0)
            else:
                method = 'Wilcoxon'
                try:
                    _, p_m = wilcoxon(residuals)
                except ValueError:
                    p_m = np.nan # In case of zero-variance residuals

            is_unbiased = p_m > 0.05 if not np.isnan(p_m) else False
            if is_unbiased: passed_bias_test += 1

            results.append({
                'Dataset_ID': name,
                'Shapiro_Stat': round(sh_stat, 4),
                'Normality_p': round(p_n, 4),
                'Is_Normal': 'Yes' if is_normal else 'No',
                'Bias_Test_Method': method,
                'Bias_p_value': round(p_m, 4) if not np.isnan(p_m) else 'N/A'
            })

        # Save findings
        result_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        result_df.to_excel(output_file, index=False)

        # Print structured analysis summary
        print("\n" + "="*60)
        print(f"[ANALYSIS] M1 Residual Evaluation Summary (n={total_groups})")
        print("-" * 60)
        print(f"Normality Adherence (p_n > 0.05): {passed_normality} groups ({(passed_normality/total_groups)*100:.2f}%)")
        print(f"Zero-Bias Adherence (p_m > 0.05): {passed_bias_test} groups ({(passed_bias_test/total_groups)*100:.2f}%)")
        print("-" * 60)
        print(f"[STATUS] Results successfully saved to: {output_file}")
        print("="*60)

    except Exception as e:
        print(f"[ERROR] An exception occurred during residual analysis: {e}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Replace with your LOOCV prediction results file path
    INPUT_FILE_PATH = r"YOUR_M1_PREDICTION_RESULTS_XLSX"
    
    # [TODO] Replace with your desired output path for the residual report
    RESIDUAL_REPORT_PATH = r"YOUR_RESIDUAL_VALIDATION_REPORT_XLSX"
    
    # -------------------------------------------------------------------------
    validate_optimized_residuals(INPUT_FILE_PATH, RESIDUAL_REPORT_PATH)