import os
import numpy as np
import pandas as pd
from scipy.stats import chi2

def calculate_bartlett_from_summary(folder_path, output_path, replicates_n=3, alpha=0.05):
    """
    Performs Bartlett's test for homogeneity of variances using summary statistics 
    (standard deviations). Validates ANOVA assumptions for experimental datasets.
    """
    if not os.path.exists(folder_path):
        print(f"[ERROR] Input directory not found: {folder_path}")
        return

    # Ensure the target directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Filter files for processing
    files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') and not f.startswith('~$')]
    
    if not files:
        print(f"[WARN] No valid .xlsx files found in: {folder_path}")
        return

    results = []
    print(f"[INFO] Initializing Bartlett's test for {len(files)} datasets...")
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        
        try:
            df = pd.read_excel(file_path)
            
            if df.empty:
                print(f"[SKIP] {file}: Dataset is empty.")
                continue

            # Extract standard deviation (SD) from the second-to-last column
            # Variance (S^2) is required for the Bartlett statistic calculation
            std_dev_col = df.iloc[:, -2]
            
            valid_variances = []
            for sd in std_dev_col:
                try:
                    sd_val = float(sd)
                    # Exclude non-positive values to avoid logarithmic errors
                    if sd_val > 0:
                        valid_variances.append(sd_val ** 2)
                except (ValueError, TypeError):
                    continue
            
            k = len(valid_variances)
            
            # Minimum requirement: at least two groups for homogeneity testing
            if k < 2:
                print(f"[WARN] {file}: Insufficient groups (k < 2). Skipping.")
                continue
            
            total_n = k * replicates_n
            
            # Calculation of Pooled Variance (Sp^2)
            # Formula: Sp^2 = Σ((n_i - 1) * S_i^2) / (N - k)
            pooled_var = sum((replicates_n - 1) * var for var in valid_variances) / (total_n - k)
            
            # Bartlett Test Statistic (T) calculation
            # Numerator: (N - k) * ln(Sp^2) - Σ((n_i - 1) * ln(S_i^2))
            numerator = (total_n - k) * np.log(pooled_var) - sum(
                (replicates_n - 1) * np.log(var) for var in valid_variances
            )
            
            # Correction factor (Denominator)
            denominator = 1 + (1 / (3 * (k - 1))) * (
                sum(1 / (replicates_n - 1) for _ in valid_variances) - (1 / (total_n - k))
            )
            
            T = numerator / denominator
            
            # P-value calculation using Chi-square distribution (df = k - 1)
            p_value = 1 - chi2.cdf(T, df=k - 1)
            
            # Assessment of Homogeneity of Variance (HOV)
            hov_met = True if p_value > alpha else False
            
            results.append({
                'Dataset': file,
                'Bartlett_Statistic': T,
                'P_value': p_value,
                'HOV_Assumed': hov_met
            })
            print(f"[STATUS] Analyzed: {file}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")

    # Export summarized results to Excel
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_excel(output_path, index=False)
        print("-" * 30)
        print(f"[STATUS] Results successfully exported to: {output_path}")
        
        # Summary statistics for manuscript verification
        passed = sum(1 for r in results if r['HOV_Assumed'])
        print(f"[SUMMARY] Total: {len(results)} | Passed HOV: {passed} | Failed HOV: {len(results) - passed}")
    else:
        print("[WARN] No valid results generated. Please verify input data format.")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Replace with your absolute directory path containing summary stats
    INPUT_FOLDER = r"YOUR_INPUT_DIRECTORY_PATH_HERE"
    
    # [TODO] Replace with your desired output file path
    OUTPUT_FILE = r"YOUR_OUTPUT_FILE_PATH_HERE\Bartlett_Test_Results.xlsx"
    
    # Analysis parameters
    SAMPLE_REPLICATES = 3 
    
    # -------------------------------------------------------------------------
    calculate_bartlett_from_summary(INPUT_FOLDER, OUTPUT_FILE, replicates_n=SAMPLE_REPLICATES)