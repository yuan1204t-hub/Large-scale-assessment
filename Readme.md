# Large-scale Assessment of Model Selection and Optimization Stability in Phytomedicine Extraction Processes

This repository provides the complete dataset, supplementary materials, and analysis workflow for a large-scale benchmark study of model selection in phytomedicine extraction-process optimization.

The study evaluates whether different candidate models provide reliable fitting performance, cross-validated predictive ability, diagnostic adequacy, and stable model-predicted optimization conclusions. The analytical framework includes traditional regression models, machine learning models, optimization-stability analysis, machine-learning-only sensitivity analysis, and stratified robustness analysis.

---

## 1. Project Overview

Phytomedicine extraction-process optimization is commonly based on small-sample experimental designs, such as Box-Behnken design, central composite design, and other response surface methodology designs. In many studies, full quadratic regression is used as the default modeling strategy. However, high training-set goodness of fit does not necessarily indicate reliable prediction or stable optimization recommendations.

This project was designed to examine model selection as a decision-oriented problem. Instead of evaluating models only by apparent fitting performance, the workflow jointly considers:

- overall fitting performance;
- leave-one-out cross-validated predictive ability;
- residual diagnostics and error stability;
- consistency of model-predicted optimal responses;
- consistency of model-predicted optimal factor combinations;
- sensitivity of conclusions in structurally excluded datasets;
- robustness of the main conclusions across dataset strata.

The main comparison includes 1,148 datasets that met the structural requirements for unified quadratic regression modeling. An additional 251 structurally excluded datasets were retained for a machine-learning-only sensitivity analysis.

---

## 2. Repository Structure

The repository is organized according to the data and supplementary materials reported in the study.

```text
Large-scale-assessment/
│
├── Additional Data (n=251)/
│   └── Structurally excluded datasets used for the machine-learning-only
│       sensitivity analysis.
│
├── Data (n=1148)/
│   └── Main analytical datasets used for the complete M0-M6 model comparison.
│
├── Figure S1. Literature screening and dataset construction.png
│
├── Figure S2. Distribution characteristics and pairwise comparisons.png
│
├── Figure S3. Model diagnostics and pairwise comparisons.png
│
├── Summary Table.xlsx
│   └── Main summary workbook containing key model-comparison results.
│
├── Table S1 Shapiro_Wilk_normality_test_overall_fitting_metrics.xlsx
│
├── Table S2 Shapiro_Wilk_predictive_ability_metrics.xlsx
│
├── Table S3 Shapiro_Wilk_diagnostic_metrics.xlsx
│
├── Table S4 GPR_supplementary_tables.xlsx
│
├── Table S5 Comparison of structural characteristics.xlsx
│
├── Table S6 Machine-learning-only sensitivity analysis.xlsx
│
├── Table S7 Stratified robustness analysis.xlsx
│
├── Table S8 dataset 1780.xlsx
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 3. Dataset Description

### 3.1 Main analytical dataset

The folder `Data (n=1148)` contains the datasets used for the complete seven-model comparison. These datasets satisfied the structural requirement for unified quadratic regression modeling.

Each dataset contains:

```text
Column 1 to Column n-1: extraction-process factors
Last column: continuous response variable
```

The main analytical dataset was used to evaluate models M0-M6 in terms of fitting performance, cross-validation performance, diagnostics, and optimization stability.

### 3.2 Additional structurally excluded dataset

The folder `Additional Data (n=251)` contains datasets that did not meet the structural requirement for full quadratic regression comparison. These datasets were not forced into the complete M0-M6 framework.

Instead, they were used for a machine-learning-only sensitivity analysis involving M3-M6. This analysis was designed to clarify the applicability boundary of the main conclusions.

---

## 4. Candidate Models

Seven candidate models were evaluated in the main analysis.

| Model | Description |
|---|---|
| M0 | Full quadratic regression model |
| M1 | Globally selected parsimonious regression model based on corrected R² |
| M2 | Globally selected parsimonious regression model based on Mallows-type Cr |
| M3 | Quadratic Ridge regression |
| M4 | Support vector regression |
| M5 | Partial least squares regression |
| M6 | Gaussian process regression |

M0-M2 were constructed within the same full quadratic candidate-term space. M3-M6 were included as machine learning models suitable for small-sample, low-dimensional, continuous-response prediction.

---

## 5. Main Analytical Workflow

The complete workflow includes the following analytical stages.

```text
Data extraction and preprocessing
        ↓
Structural eligibility assessment
        ↓
Complete M0-M6 model comparison for 1,148 datasets
        ↓
Overall fitting-performance analysis
        ↓
Leave-one-out cross-validation analysis
        ↓
Model diagnostic analysis
        ↓
Optimization-stability analysis
        ↓
Machine-learning-only sensitivity analysis for 251 excluded datasets
        ↓
Stratified robustness analysis
```

---

## 6. Main Analysis

The main analysis was conducted on the 1,148 datasets in `Data (n=1148)`.

It includes:

- fitting M0-M6 for each dataset;
- calculating fitting metrics, including R², RMSE, and MAE;
- calculating leave-one-out cross-validation metrics, including Q², RMSECV, and MAECV;
- evaluating diagnostic metrics, including residual normality, residual mean deviation, maximum absolute residual, and AE-IQR;
- identifying model-predicted optimal responses and optimal factor combinations;
- comparing optimization stability between the three-model and seven-model settings.

The main summary results are provided in:

```text
Summary Table.xlsx
```

---

## 7. Machine-learning-only Sensitivity Analysis

The machine-learning-only sensitivity analysis was conducted on the 251 structurally excluded datasets in:

```text
Additional Data (n=251)/
```

These datasets were excluded from the complete M0-M6 comparison because they did not provide sufficient structural support for estimating the full quadratic regression model. However, they were still informative for examining whether the main model-applicability pattern changed in structurally limited datasets.

The sensitivity analysis evaluated:

- M3: quadratic Ridge regression;
- M4: support vector regression;
- M5: partial least squares regression;
- M6: Gaussian process regression.

The main metrics were:

- Q²;
- RMSECV;
- MAECV.

The corresponding supplementary table is:

```text
Table S6 Machine-learning-only sensitivity analysis.xlsx
```

A recommended script name for reproducing this part is:

```text
run_ml_sensitivity_analysis.py
```

---

## 8. Stratified Robustness Analysis

The stratified robustness analysis was conducted to examine whether the main conclusions were consistent across different dataset structures.

The analysis summarized model performance and optimization-stability results by:

- experimental design type;
- sample-size group;
- number-of-factors group.

The stratified analysis includes:

- subgroup sample counts;
- model-level Q², RMSECV, MAECV, and AE-IQR summaries;
- model rankings within each subgroup;
- three-model and seven-model Response CV summaries;
- three-model and seven-model Average Factor CV summaries;
- compact tables for reporting stratified robustness in the manuscript.

The corresponding supplementary table is:

```text
Table S7 Stratified robustness analysis.xlsx
```

A recommended script name for reproducing this part is:

```text
run_stratified_robustness_summary.py
```

---

## 9. Supplementary Tables and Figures

### 9.1 Figures

| File | Description |
|---|---|
| Figure S1 | Literature screening, dataset exclusion, and analytical-set construction |
| Figure S2 | Supplementary model-comparison results for traditional regression models |
| Figure S3 | Supplementary diagnostic results and pairwise comparisons |

### 9.2 Supplementary tables

| File | Description |
|---|---|
| Table S1 | Shapiro-Wilk normality test results for fitting metrics |
| Table S2 | Shapiro-Wilk normality test results for predictive metrics |
| Table S3 | Shapiro-Wilk normality test results for diagnostic metrics |
| Table S4 | Supplementary GPR analysis |
| Table S5 | Comparison of structural characteristics between included and excluded datasets |
| Table S6 | Machine-learning-only sensitivity analysis |
| Table S7 | Stratified robustness analysis |
| Table S8 | Worked-example dataset 1780 |

---

## 10. Installation

Python 3.8 or later is recommended.

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

Main Python packages used in this project include:

- pandas
- numpy
- scipy
- scikit-learn
- statsmodels
- openpyxl
- matplotlib

The complete package list is provided in:

```text
requirements.txt
```

---

## 11. Usage

Before running the scripts, update the input and output paths in the user settings section of each script.

A typical workflow is:

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the main model-comparison workflow
python run_main_model_comparison.py

# Step 3: Run the machine-learning-only sensitivity analysis
python run_ml_sensitivity_analysis.py

# Step 4: Run the stratified robustness summary
python run_stratified_robustness_summary.py
```

Script names may differ depending on how the analysis files are organized. The key requirement is that the input folders and output paths should be correctly specified before running each script.

---

## 12. Expected Output Files

The scripts generate Excel workbooks and figures, including:

- dataset-level model results;
- model-level summary statistics;
- cross-validation result tables;
- diagnostic result tables;
- optimization-stability summaries;
- best-model count summaries;
- stratified robustness summaries;
- supplementary figures and tables.

Most outputs are saved as `.xlsx` files, with figures saved as image files.

---

## 13. Reproducibility Notes

To reproduce the analysis, please ensure that:

1. all datasets are formatted with predictors in the first columns and the response variable in the final column;
2. the dataset folders are not mixed with previous result files;
3. path settings in each script are updated before execution;
4. required Python packages are installed;
5. intermediate outputs from earlier analytical steps are available before running summary scripts.

The stratified robustness script does not refit models. It summarizes model outputs that have already been generated. The machine-learning sensitivity analysis script refits M3-M6 models for the structurally excluded datasets.

---

## 14. Citation

If you use this repository or adapt the workflow in your own research, please cite the associated study or archived code package.

```text
Tao, Y. (2026). Large-scale assessment of model selection and optimization stability in phytomedicine extraction processes. Zenodo.
```

If the related manuscript has been formally published, please cite the final journal article instead of or in addition to the archived code package.

---

## 15. License

This project is licensed under the MIT License. See the `LICENSE` file for details.