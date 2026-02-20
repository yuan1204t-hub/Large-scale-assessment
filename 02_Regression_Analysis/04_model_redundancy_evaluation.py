import os
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

def calculate_pmax_and_insignificant_ratio(input_dir, output_file):
    """
    Evaluates the full quadratic models (M0) to extract the maximum p-value (p_max)
    and the proportion of statistically insignificant regression terms (rho).
    This highlights the over-parameterization issue in standard RSM practices.
    """
    if not os.path.exists(input_dir):
        print(f"[ERROR] Directory not found: {input_dir}")
        return

    # Fetch valid Excel files
    files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
    if not files:
        print(f"[WARN] No valid .xlsx files found in: {input_dir}")
        return

    print(f"[INFO] Evaluating model redundancy (p_max & rho) for {len(files)} datasets...")

    results = []
    total_files = len(files)
    
    # Counters for summary statistics (aligns with manuscript findings)
    all_significant_count = 0  # Models where p_max < 0.05
    high_redundancy_count = 0  # Models where rho > 0.5

    for file in files:
        filepath = os.path.join(input_dir, file)
        
        try:
            # Read data from the standardized worksheet
            df = pd.read_excel(filepath, sheet_name='Before')
            
            if df.empty:
                print(f"[SKIP] {file}: Dataset is empty.")
                continue

            X = df.iloc[:, :-1]
            y = df.iloc[:, -1].astype(float)

            # Generate quadratic features (without bias to avoid collinearity with sm.add_constant)
            pf = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = pf.fit_transform(X)
            
            # Add constant for OLS fitting
            X_with_const = sm.add_constant(X_poly)
            
            # Fit OLS model
            model = sm.OLS(y, X_with_const).fit()
            p_values = model.pvalues
            
            # Calculate p_max and rho (proportion of insignificant terms)
            p_max = p_values.max()
            insignificant_terms = p_values[p_values > 0.05]
            
            insignificant_count = len(insignificant_terms)
            total_terms = len(p_values)
            rho = insignificant_count / total_terms
            
            # Update summary counters
            if p_max < 0.05:
                all_significant_count += 1
            if rho > 0.5:
                high_redundancy_count += 1

            # Append structured results
            results.append({
                'Dataset': file,
                'p_max': round(p_max, 4),
                'Insignificant_Count': insignificant_count,
                'Redundancy_Ratio_rho': round(rho, 4)
            })
            print(f"[STATUS] Analyzed: {file}")

        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")

    # Save aggregated results
    if results:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        results_df = pd.DataFrame(results)
        results_df.to_excel(output_file, index=False)
        
        # Print summary statistics mimicking the manuscript narrative
        print("-" * 60)
        print(f"[SUMMARY] Model Redundancy Diagnostics:")
        print(f"Total Datasets Processed : {total_files}")
        print(f"All Terms Significant (p_max < 0.05) : {all_significant_count} ({(all_significant_count/total_files)*100:.2f}%)")
        print(f"High Redundancy (rho > 0.5)           : {high_redundancy_count} ({(high_redundancy_count/total_files)*100:.2f}%)")
        print(f"[INFO] Detailed results exported to: {output_file}")
        print("-" * 60)
    else:
        print("[WARN] No results generated. Please verify input data.")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Replace with your absolute directory path for RSM source data
    INPUT_DIR = r"YOUR_RSM_DATA_DIRECTORY_PATH_HERE"
    
    # [TODO] Replace with your desired output path for the redundancy report
    OUTPUT_PATH = r"YOUR_OUTPUT_REPORT_PATH_HERE\RSM_Model_Redundancy_Report.xlsx"
    
    # -------------------------------------------------------------------------
    calculate_pmax_and_insignificant_ratio(INPUT_DIR, OUTPUT_PATH)