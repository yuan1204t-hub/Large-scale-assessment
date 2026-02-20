import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

def evaluate_residual_validity(input_dir, output_file):
    """
    Evaluates the statistical validity of residuals for the full quadratic model.
    Includes Adjusted R-squared, Shapiro-Wilk (normality), and Mean Bias Test.
    """
    if not os.path.exists(input_dir):
        print(f"[ERROR] Directory not found: {input_dir}")
        return

    # Scan for valid Excel files
    files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
    if not files:
        print(f"[WARN] No valid .xlsx files found in: {input_dir}")
        return

    results = []
    normality_passed = 0   
    bias_insignificant = 0 
    total_groups = len(files)

    print(f"[INFO] Evaluating residual diagnostics for {total_groups} RSM datasets...")

    for file in files:
        filepath = os.path.join(input_dir, file)
        
        try:
            # Data extraction from the pre-encoded experimental sheet '编码后'
            df = pd.read_excel(filepath, sheet_name='编码后')
            
            if df.empty:
                print(f"[SKIP] {file}: Dataset is empty.")
                continue

            X = df.iloc[:, :-1]
            y = df.iloc[:, -1].astype(float)

            # Feature engineering: Full quadratic expansion (M0)
            pf = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = pf.fit_transform(X)
            X_with_const = sm.add_constant(X_poly)
            
            # Ordinary Least Squares (OLS) regression and residual computation
            model = sm.OLS(y, X_with_const).fit()
            adj_r2 = model.rsquared_adj
            residuals = model.resid

            # 1. Normality Assessment (Shapiro-Wilk)
            _, p_n = stats.shapiro(residuals)
            if p_n >= 0.05:
                normality_passed += 1

            # 2. Unbiasedness Assessment (Mean Zero-Bias)
            # Use parametric T-test if normal, else non-parametric Wilcoxon
            if p_n > 0.05:
                _, p_m = stats.ttest_1samp(residuals, 0)
            else:
                _, p_m = stats.wilcoxon(residuals)

            if p_m >= 0.05:
                bias_insignificant += 1

            results.append({
                'Dataset': file,
                'Shapiro_Wilk_pn': round(p_n, 4),
                'Mean_Bias_pm': round(p_m, 4),
                'Adjusted_R2': round(adj_r2, 4)
            })
            print(f"[STATUS] Diagnosed: {file}")

        except Exception as e:
            print(f"[ERROR] Failed to diagnose {file}: {e}")

    # Export summarized diagnostic results
    if results:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        pd.DataFrame(results).to_excel(output_file, index=False)
        
        print("-" * 60)
        print(f"[SUMMARY] Diagnostic report finalized.")
        print(f"[RESIDUALS] Normality Adherence (p_n >= 0.05): {normality_passed}/{len(results)}")
        print(f"[BIAS] Mean Unbiasedness (p_m >= 0.05): {bias_insignificant}/{len(results)}")
        print(f"[INFO] Report Location: {output_file}")
        print("-" * 60)
    else:
        print("[WARN] No diagnostic results generated. Please check input data.")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Replace with your absolute directory path for RSM datasets
    INPUT_DIR = r"YOUR_RSM_DATA_DIRECTORY_PATH_HERE"
    
    # [TODO] Replace with your desired output path for the diagnostic report
    OUTPUT_PATH = r"YOUR_OUTPUT_REPORT_PATH_HERE\Quadratic_Residual_Diagnostics_RSM.xlsx"
    
    # -------------------------------------------------------------------------
    evaluate_residual_validity(INPUT_DIR, OUTPUT_PATH)