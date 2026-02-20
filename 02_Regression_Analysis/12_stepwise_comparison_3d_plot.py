import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ==============================================================================
# Script: 12_stepwise_comparison_3d_plot.py
# Description: Generates a 3D scatter plot to visualize the discrepancies in 
#              R2 values across Matlab, Python, and R platforms.
#              The deviation from the x=y=z line highlights algorithmic 
#              inconsistency discussed in Section 3.1.
# ==============================================================================

def plot_3d_software_comparison(file_path, save_path):
    """
    Reads the aggregated software results and creates a 3D visualization
    of R-squared values to compare platform reproducibility.
    """
    print("正在读取数据并准备生成 3D 散点图...")

    if not os.path.exists(file_path):
        print(f"[错误] 找不到汇总文件: {file_path}")
        return

    try:
        # Load the aggregated results (assuming header is in the first row)
        df = pd.read_excel(file_path)
        
        # Clean data: Ensure columns exist and convert to float, dropping NaNs
        # Based on '11_cross_platform_aggregation.py' output columns
        cols = ['MATLAB_R2', 'Python_R2', 'R_R2']
        plot_df = df[cols].apply(pd.to_numeric, errors='coerce').dropna()
        
        matlab_r2 = plot_df['MATLAB_R2'].values
        python_r2 = plot_df['Python_R2'].values
        r_r2 = plot_df['R_R2'].values

        if len(matlab_r2) == 0:
            print("[错误] 无有效数据可供绘制。")
            return

        # Initialize the 3D figure
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plotting the three perspectives to show symmetry/divergence
        # Red: (Matlab, Python, R)
        ax.scatter(matlab_r2, python_r2, r_r2, c='red', s=40, alpha=0.6, label='Matlab View')
        # Green: (Python, R, Matlab)
        ax.scatter(python_r2, r_r2, matlab_r2, c='green', s=40, alpha=0.6, label='Python View')
        # Blue: (R, Matlab, Python)
        ax.scatter(r_r2, matlab_r2, python_r2, c='blue', s=40, alpha=0.6, label='R View')

        # Add x = y = z reference line (The "Perfect Agreement" line)
        all_vals = np.concatenate([matlab_r2, python_r2, r_r2])
        min_v, max_v = all_vals.min(), all_vals.max()
        line = np.linspace(min_v, max_v, 100)
        ax.plot(line, line, line, 'k--', linewidth=2, label='Identity Line (x=y=z)')

        # Axis labeling and styling
        ax.set_xlabel('Matlab $R^2$', fontsize=12)
        ax.set_ylabel('Python $R^2$', fontsize=12)
        ax.set_zlabel('R $R^2$', fontsize=12)
        ax.set_title('3D Distribution of $R^2$ Values Across Platforms', fontsize=15, pad=20)

        # Legend configuration
        ax.legend(loc='upper left', fontsize=10, frameon=False)

        # Adjust layout and save with high resolution for publication
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        
        print(f"🎉 3D 对比图已成功生成！保存位置：\n -> {save_path}")

    except Exception as e:
        print(f"[错误] 绘图过程中发生异常: {e}")

if __name__ == "__main__":
    # Define file paths
    SUMMARY_FILE = r"phytomedicine/数据/软件汇总.xlsx"
    PLOT_OUTPUT = r"phytomedicine/数据/r2_3d_plot.png"
    
    plot_3d_software_comparison(SUMMARY_FILE, PLOT_OUTPUT)