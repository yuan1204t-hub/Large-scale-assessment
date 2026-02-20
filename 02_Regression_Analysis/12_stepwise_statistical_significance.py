import os
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg

# ==============================================================================
# Script: 12_stepwise_statistical_significance.py
# Description: Conducts comprehensive statistical testing (Normality, HOV, ANOVA)
#              to determine if R2 differences across platforms are significant.
# ==============================================================================

def perform_r2_significance_testing(file_path):
    """
    Executes a statistical pipeline to compare R2 distributions:
    1. Shapiro-Wilk test for Normality.
    2. Levene's test for Homogeneity of Variance (HOV).
    3. One-way ANOVA or Welch's ANOVA.
    4. Post-hoc testing (Tukey HSD or Games-Howell).
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] Target file not found: {file_path}")
        return

    print(f"[INFO] Initializing statistical significance analysis: {os.path.basename(file_path)}")

    try:
        # Load the software aggregation summary
        df = pd.read_excel(file_path)
        
        # Clean data: drop missing values for specific columns
        # Ensuring column names match the aggregated summary output
        data_mat = df['MATLAB_R2'].dropna()
        data_py = df['Python_R2'].dropna()
        data_r = df['R_R2'].dropna()

        # --- 1. Normality Test (Shapiro-Wilk) ---
        print("\n" + "="*30)
        print("[ANALYSIS] 1. Normality Assessment (Shapiro-Wilk)")
        norm_results = []
        for label, data in zip(['MATLAB', 'Python', 'R'], [data_mat, data_py, data_r]):
            stat, p = stats.shapiro(data)
            is_normal = p > 0.05
            norm_results.append(is_normal)
            print(f"  {label:8}: W = {stat:.4f}, p = {p:.4e} -> {'✔ Normal' if is_normal else '✘ Non-normal'}")

        # --- 2. Homogeneity of Variance Test (Levene's) ---
        print("\n[ANALYSIS] 2. Homogeneity of Variance (Levene's)")
        levene_stat, levene_p = stats.levene(data_mat, data_py, data_r)
        is_homogenous = levene_p > 0.05
        print(f"  Levene Stat = {levene_stat:.4f}, p = {levene_p:.4e} -> {'✔ Homogenous' if is_homogenous else '✘ Heteroscedastic'}")

        # Reshape data to "Long Format" for ANOVA and Post-hoc tests
        df_long = pd.melt(df, value_vars=['MATLAB_R2', 'Python_R2', 'R_R2'], 
                         var_name='group', value_name='R2').dropna()

        # --- 3. One-way ANOVA Analysis ---
        print("\n[ANALYSIS] 3. Variance Analysis (ANOVA)")
        significant_diff = False
        p_anova = 1.0

        if is_homogenous:
            # Traditional ANOVA if variances are equal
            anova_res = stats.f_oneway(data_mat, data_py, data_r)
            p_anova = anova_res.pvalue
            significant_diff = p_anova < 0.05
            print(f"  [EXEC] One-way ANOVA performed (HOV met):")
            print(f"  F-stat = {anova_res.statistic:.4f}, p = {p_anova:.4e}")
        else:
            # Welch's ANOVA if variances are unequal
            welch_res = pg.welch_anova(dv='R2', between='group', data=df_long)
            p_anova = welch_res['p-unc'].values[0]
            significant_diff = p_anova < 0.05
            print(f"  [EXEC] Welch's ANOVA performed (HOV not met):")
            print(welch_res)

        # --- 4. Pairwise Comparisons & Post-hoc ---
        print("\n" + "="*30)
        print("[ANALYSIS] 4. Pairwise Robust t-tests (Welch's)")
        pairs = [('MATLAB', 'Python', data_mat, data_py), 
                 ('MATLAB', 'R', data_mat, data_r), 
                 ('Python', 'R', data_py, data_r)]
        
        for name1, name2, d1, d2 in pairs:
            t_stat, tp = stats.ttest_ind(d1, d2, equal_var=False)
            sig_mark = ' (p < 0.05) *' if tp < 0.05 else ' (n.s.)'
            print(f"  {name1:8} vs {name2:8}: p = {tp:.10f}{sig_mark}")

        if significant_diff:
            print("\n[STATUS] Global significance detected. Proceeding to Post-hoc Testing:")
            if is_homogenous:
                print("  [EXEC] Tukey HSD (Parametric)")
                tukey = pairwise_tukeyhsd(endog=df_long['R2'], groups=df_long['group'], alpha=0.05)
                print(tukey.summary())
            else:
                print("  [EXEC] Games-Howell (Non-parametric robust)")
                gh = pg.pairwise_gameshowell(dv='R2', between='group', data=df_long)
                print(gh)
        else:
            print("\n[STATUS] No significant difference detected across platforms (p >= 0.05).")
        
        print("="*30)
        print("[COMPLETE] Statistical pipeline finished.")

    except Exception as e:
        print(f"[ERROR] An exception occurred during analysis: {e}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify This Path)
    # =========================================================================
    # [TODO] Replace with the path to your aggregated cross-platform summary file
    AGGREGATED_SUMMARY = r"YOUR_SUMMARY_FILE_PATH_HERE"
    
    # -------------------------------------------------------------------------
    perform_r2_significance_testing(AGGREGATED_SUMMARY)