import pandas as pd
import os

# ==============================================================================
# Script: 05_cv_stability_classification_stats.py
# Section: 3.3 Stability Assessment
# Description: Categorizes Coefficient of Variation (CV) results into 
#              stability levels: High (<10%), Moderate (10-30%), and Low (>30%).
#              Generates quantitative distribution statistics for the manuscript.
# ==============================================================================

def analyze_cv_distribution(input_file):
    """
    Classifies datasets based on Response CV thresholds to evaluate 
    process recommendation stability.
    """
    if not os.path.exists(input_file):
        print(f"[ERROR] Summary CV file not found: {input_file}")
        return

    print(f"[INFO] Initializing Stability Classification Analysis...")

    try:
        # 1. Load CV data (Reads primary worksheet)
        df = pd.read_excel(input_file)
        
        # Identify the CV column (Supporting both English and Chinese headers)
        col_name = next((c for c in df.columns if 'Response_CV' in str(c) or '因变量CV' in str(c)), None)
        
        if not col_name:
            print(f"[ERROR] Required CV column not found. Headers: {df.columns.tolist()}")
            return

        cv_values = pd.to_numeric(df[col_name], errors='coerce').dropna()
        total_n = len(cv_values)

        if total_n == 0:
            print("[WARN] No valid numeric CV values found for analysis.")
            return

        # 2. Classification Logic (User-defined thresholds)
        # Thresholds: <10% (High), 10-30% (Moderate), >30% (Low)
        count_high = (cv_values < 0.10).sum()
        count_moderate = ((cv_values >= 0.10) & (cv_values <= 0.30)).sum()
        count_low = (cv_values > 0.30).sum()

        # 3. Calculation of Proportions
        pct_high = (count_high / total_n) * 100
        pct_mod = (count_moderate / total_n) * 100
        pct_low = (count_low / total_n) * 100

        # 4. Results Aggregation
        stats_summary = pd.DataFrame({
            'Stability_Level': ['High Stability (<10%)', 'Moderate Stability (10-30%)', 'Low Stability (>30%)'],
            'Count': [count_high, count_moderate, count_low],
            'Percentage': [f"{pct_high:.2f}%", f"{pct_mod:.2f}%", f"{pct_low:.2f}%"]
        })

        # --- Terminal Report ---
        print("\n" + "="*60)
        print("STABILITY CLASSIFICATION REPORT (Response CV)")
        print("-" * 60)
        print(stats_summary.to_string(index=False))
        print("-" * 60)
        print(f"Total Datasets Analyzed: {total_n}")
        print("="*60)

        # 5. Interpretive Note for Manuscript Discussion
        print("\n[ANALYSIS] Discussion Insights:")
        if pct_high > 80:
            print(" -> Result: The model exhibits EXCELLENT stability for process advice.")
        elif pct_high > 50:
            print(" -> Result: The model demonstrates GOOD robustness with minor variability.")
        else:
            print(" -> Result: Stability is VARIABLE; consider investigating factor interactions.")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during classification: {e}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    
    # [TODO] Path to the aggregated CV results (Output from Script 04)
    STABILITY_SUMMARY_XLS = r"YOUR_PATH_TO_CV_SUMMARY_FILE.xlsx"

    # -------------------------------------------------------------------------
    analyze_cv_distribution(STABILITY_SUMMARY_XLS)