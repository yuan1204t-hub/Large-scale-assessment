import os
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

# ==============================================================================
# Script: 07_full_quadratic_model_evaluation.py
# Description: Evaluates the Full Quadratic Model (M0). 
#              Represents the standard RSM approach without variable selection.
#              Used as a benchmark to compare against optimized models.
# ==============================================================================

def evaluate_full_quadratic_models(input_folder, output_path):
    """
    Batch processes datasets to fit full quadratic models and extract
    goodness-of-fit and significance metrics.
    """
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input directory not found: {input_folder}")
        return

    print(f"[INFO] Initializing Batch Evaluation for Full Quadratic (M0) Models...")

    results = []
    # Fetch valid Excel files
    files = [f for f in os.listdir(input_folder) if f.endswith(".xlsx") and not f.startswith("~$")]
    
    if not files:
        print(f"[WARN] No valid .xlsx files found in: {input_folder}")
        return

    print(f"[INFO] Found {len(files)} datasets. Starting regression analysis...")

    for filename in files:
        file_path = os.path.join(input_folder, filename)
        try:
            # Read data (Defaults to the first worksheet)
            df = pd.read_excel(file_path)
            
            # Ensure numeric data and drop missing values
            df = df.apply(pd.to_numeric, errors='coerce').dropna()

            # Separate Features (X) and Response (y)
            X_raw = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            # 1. Quadratic Expansion
            # include_bias=False to handle intercept via statsmodels add_constant
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly_values = poly.fit_transform(X_raw)
            feature_names = poly.get_feature_names_out(X_raw.columns)
            X_poly_df = pd.DataFrame(X_poly_values, columns=feature_names)

            # 2. Fit OLS Model (with Intercept)
            X_with_const = sm.add_constant(X_poly_df, has_constant='add')
            model = sm.OLS(y, X_with_const).fit()

            # 3. Extract Performance Metrics
            r2 = model.rsquared
            adj_r2 = model.rsquared_adj
            max_p = model.pvalues.max()

            results.append({
                "Dataset_ID": filename,
                "R2": round(r2, 4),
                "Adjusted_R2": round(adj_r2, 4),
                "Max_P_Value": round(max_p, 4),
                "Total_Terms_Nt": len(model.params) - 1  # Excluding intercept
            })
            print(f"[STATUS] Analyzed: {filename} | Nt: {len(model.params)-1} | max_p: {max_p:.4f}")

        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")

    # Export summarized results
    if results:
        results_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_excel(output_path, index=False)
        print("-" * 60)
        print(f"[COMPLETE] Full Quadratic (M0) evaluation finished.")
        print(f"[INFO] Summary report generated at: {output_path}")
        print("-" * 60)
    else:
        print("[WARN] No results were generated. Please check input data.")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Replace with your absolute directory path for M0 datasets
    INPUT_DIR_PATH = r"YOUR_INPUT_DIRECTORY_PATH_HERE"
    
    # [TODO] Replace with your desired output path for the M0 evaluation report
    OUTPUT_REPORT_PATH = r"YOUR_OUTPUT_REPORT_PATH_HERE\M0_Full_Quadratic_Summary.xlsx"
    
    # -------------------------------------------------------------------------
    evaluate_full_quadratic_models(INPUT_DIR_PATH, OUTPUT_REPORT_PATH)