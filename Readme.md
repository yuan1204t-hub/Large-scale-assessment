# Large-scale Assessment of Regression Modeling Practices in Phytomedicine Extraction Process Optimization

This repository provides the complete Python-based analysis workflow for a large-scale assessment of regression modeling practices in phytomedicine extraction process optimization. The framework was designed to evaluate the explanatory performance, predictive reliability, diagnostic validity, and optimization stability of different regression modeling strategies commonly used in extraction process studies.

The repository includes scripts for data preprocessing, regression model evaluation, global model optimization, cross-validation analysis, residual diagnostics, and stability assessment of model-predicted optimal extraction conditions.

---

## 1. Project Overview

Phytomedicine extraction process optimization is commonly based on small-sample experimental designs, such as Box-Behnken design, central composite design, and other response surface methodology designs. In many studies, the full quadratic response surface model is used as the default modeling strategy. However, a model with better apparent fitting performance does not necessarily provide better predictive reliability or more stable optimization conclusions.

This project aims to compare multiple candidate modeling strategies under a unified evaluation framework. The workflow includes:

- data extraction and preprocessing;
- regression model construction and comparison;
- model-level and coefficient-level statistical evaluation;
- leave-one-out cross-validation;
- residual diagnostic analysis;
- global optimization of model-predicted extraction conditions;
- stability assessment of predicted optimal responses and factor combinations.

---

## 2. Repository Structure

The repository is organized according to the main analytical stages of the study.

```text
Large-scale-assessment/
│
├── 01_Data_Preprocessing/
│   └── Scripts for extracting, cleaning, and preparing datasets.
│
├── 02_Regression_Analysis/
│   └── Scripts for model fitting, model comparison, and statistical evaluation.
│
├── 03_Global_Optimization/
│   └── Scripts for model selection, cross-validation, diagnostic analysis,
│       and global optimization.
│
├── 04_Stability_Analysis/
│   └── Scripts for evaluating the stability of predicted optimal responses
│       and optimal factor combinations.
│
├── Data/
│   └── Data.zip
│       Complete dataset used in this study.
│
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## 3. Main Analytical Modules

### 3.1 Data Preprocessing

The scripts in `01_Data_Preprocessing` are used to prepare the raw experimental datasets for batch analysis.

Main functions include:

- reading Excel files in batches;
- extracting independent variables and response variables;
- checking data structure and missing values;
- standardizing file formats for subsequent regression analysis.

By default, each dataset should be arranged as follows:

```text
Column 1 to Column n-1: independent variables
Last column: response variable
```

---

### 3.2 Regression Analysis

The scripts in `02_Regression_Analysis` and related subfolders perform regression model fitting and statistical evaluation.

This part includes:

- construction of full quadratic regression models;
- extraction of model-level evaluation metrics;
- summary of fitting performance;
- comparison of alternative candidate models;
- normality tests and significance tests for model comparison.

Typical metrics include:

- coefficient of determination;
- adjusted or modified coefficient of determination;
- root mean squared error;
- mean absolute error;
- model-level p-value;
- coefficient-level p-values;
- number of selected predictors.

---

### 3.3 Cross-validation and Diagnostic Evaluation

The cross-validation and diagnostic scripts evaluate whether the fitted models have reliable predictive ability and acceptable residual behavior.

This module includes:

- leave-one-out cross-validation;
- calculation of Q², RMSECV, and MAECV;
- Shapiro-Wilk normality testing;
- Friedman tests and Wilcoxon paired comparisons;
- residual bias testing;
- Cook's distance analysis;
- maximum error and maximum absolute error evaluation.

The purpose of this module is to avoid relying only on apparent goodness-of-fit and to provide a more comprehensive assessment of model reliability.

---

### 3.4 Global Optimization and Stability Analysis

The scripts in `03_Global_Optimization` and `04_Stability_Analysis` are used to estimate the model-predicted optimal extraction conditions and evaluate the stability of optimization conclusions.

This module includes:

- global grid search within the experimental factor space;
- identification of predicted optimal response values;
- identification of optimal factor combinations;
- comparison of optimal conditions across different models;
- calculation of coefficient of variation for predicted optimal responses;
- assessment of the consistency of model-based process recommendations.

The same search space and grid resolution are applied to all candidate models to ensure comparability.

---

## 4. Model Types

The study compares multiple regression modeling strategies. The main model categories include:

- `M0`: full quadratic regression model;
- `M1`: globally selected regression subset model prioritizing explanatory adequacy;
- `M2`: globally selected regression subset model prioritizing structural parsimony;
- `M3-M6`: additional candidate models or machine-learning-based comparison models, depending on the specific analytical section.

The exact scripts for each model are provided in the corresponding folders.

---

## 5. Dataset Availability

The complete dataset supporting this study is archived in:

```text
Data/Data.zip
```

To reproduce the analysis:

1. Download or clone this repository.
2. Extract `Data.zip`.
3. Update the input and output paths in the relevant Python scripts.
4. Run the scripts in the recommended order.

To analyze your own data, please format each dataset as an Excel file with independent variables in the first columns and the response variable in the last column.

---

## 6. Installation

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

The complete package list is provided in `requirements.txt`.

---

## 7. Usage

Because many scripts rely on intermediate outputs generated by earlier steps, the analysis should be performed in the following order:

```text
01_Data_Preprocessing
        ↓
02_Regression_Analysis
        ↓
03_Global_Optimization
        ↓
04_Stability_Analysis
```

Before running each script, please check the path configuration section, usually located near the end of the file:

```python
if __name__ == "__main__":
    # Path configuration
    input_dir = r"..."
    output_dir = r"..."
```

You should replace these paths with the corresponding directories on your own computer.

---

## 8. Recommended Workflow

A typical workflow is:

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Extract the dataset
unzip Data/Data.zip -d Data/

# Step 3: Run preprocessing scripts
python 01_Data_Preprocessing/your_script_name.py

# Step 4: Run regression analysis scripts
python 02_Regression_Analysis/your_script_name.py

# Step 5: Run global optimization scripts
python 03_Global_Optimization/your_script_name.py

# Step 6: Run stability analysis scripts
python 04_Stability_Analysis/your_script_name.py
```

Please note that script names may vary according to the specific analysis task.

---

## 9. Output Files

The scripts generate several types of output files, including:

- model metric summary tables;
- cross-validation result tables;
- residual diagnostic result tables;
- model comparison statistics;
- optimal response summaries;
- optimal factor combination summaries;
- figures for model comparison and diagnostic visualization.

Most outputs are saved as Excel files or image files in the user-defined output directories.

---

## 10. Citation

If you use this repository or adapt the workflow in your own research, please cite the associated study or archived code package:

```text
Tao, Y. (2026). Large-scale assessment of regression modeling practices in phytomedicine extraction process optimization. Zenodo. https://doi.org/10.5281/zenodo.18712471
```

If the related manuscript has been formally published, please cite the final journal article instead of or in addition to the archived code package.

---

## 11. License

This project is licensed under the MIT License. See the `LICENSE` file for details.
****
