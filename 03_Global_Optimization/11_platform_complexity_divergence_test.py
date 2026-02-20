import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg
import os

# ==============================================================================
# Script: 10_platform_complexity_divergence_test.py
# Section: 3.2 Optimal Modeling Strategy
# Description: Compares the number of included variables (model complexity, Nt) 
#              across Matlab, Python, and R. Evaluates if standard stepwise 
#              algorithms lead to structural divergence across platforms.
# ==============================================================================

def analyze_platform_complexity(file_path):
    """
    Executes a statistical comparison of model terms (Nt) across platforms.
    Logic: Levene's Test -> ANOVA/Welch -> Tukey/Games-Howell.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] Input file not found: {file_path}")
        return

    print(f"[INFO] Starting consistency analysis for model complexity ($N_t$)...")

    try:
        # 1. Load and Clean Data
        df = pd.read_excel(file_path)
        
        # Melt to long format for categorical statistical testing
        # Using the column names provided in your logic
        df_long = pd.melt(df, 
                          value_vars=['Matlab', 'Python', 'R'],
                          var_name='Platform', 
                          value_name='Nt').dropna()

        # 2. Descriptive Statistics
        summary = df_long.groupby('Platform')['Nt'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(f"\n[ANALYSIS] Descriptive Statistics for Model Terms ($N_t$):")
        print(summary)

        # 3. Homogeneity of Variance (Levene's Test)
        _, p_levene = stats.levene(
            df['Matlab'].dropna(), 
            df['Python'].dropna(), 
            df['R'].dropna()
        )
        is_homo = p_levene > 0.05
        print(f"\n[STATUS] Levene's Test for Variance Homogeneity: p = {p_levene:.4e}")
        print(f"[STATUS] Variances are {'Homogeneous' if is_homo else 'Heteroscedastic'}")

        # 4. Global Significance Testing
        p_global = 1.0
        if is_homo:
            # Standard One-way ANOVA for equal variances
            res = stats.f_oneway(
                df['Matlab'].dropna(), 
                df['Python'].dropna(), 
                df['R'].dropna()
            )
            p_global = res.pvalue
            test_name = "One-way ANOVA"
        else:
            # Welch's ANOVA for unequal variances
            res = pg.welch_anova(dv='Nt', between='Platform', data=df_long)
            p_global = res['p-unc'].values[0]
            test_name = "Welch's ANOVA"

        print(f"[STATUS] Global Significance Result ({test_name}): p = {p_global:.4e}")

        # 5. Post-hoc Multiple Comparison
        if p_global < 0.05:
            print(f"[RESULT] Significant Difference: Platform-specific structural divergence detected.")
            if is_homo:
                print(f"[INFO] Performing Tukey HSD Post-hoc Test...")
                posthoc = pairwise_tukeyhsd(df_long['Nt'], df_long['Platform'])
                print(posthoc.summary())
            else:
                print(f"[INFO] Performing Games-Howell Post-hoc Test...")
                print(pg.pairwise_gameshowell(dv='Nt', between='Platform', data=df_long))
        else:
            print(f"[RESULT] No Significant Difference: Complexity is relatively consistent across platforms.")

    except Exception as e:
        print(f"[ERROR] Statistical analysis failed: {e}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    
    # [TODO] Replace with your absolute path to the aggregated results file
    INPUT_FILE_PATH = r"YOUR_SOFTWARE_AGGREGATION_FILE_PATH_HERE.xlsx"

    # -------------------------------------------------------------------------
    analyze_platform_complexity(INPUT_FILE_PATH)