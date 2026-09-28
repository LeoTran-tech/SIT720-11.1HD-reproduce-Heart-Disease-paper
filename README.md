# SIT720 11.1HD - Machine Learning Mini Research

## Project title

Reproduction and Leakage-Aware Evaluation of a Stacking-Based Heart Attack Prediction Model

## Overview

This project reproduces the machine-learning experiments from:

M. Bhagat, A. Sharma, and P. Agarwal, "An efficient stacking-based ensemble technique for early heart attack prediction," Multimedia Tools and Applications, vol. 84, pp. 36351-36375, 2025. DOI: 10.1007/s11042-024-19293-7.

Part 1 reproduces the paper's Logistic Regression, Decision Tree, Random Forest, XGBoost, Naive Bayes, K-Nearest Neighbours, and 5-fold stacking experiments.

Part 2 audits the dataset for duplicate records and evaluates a leakage-aware pipeline using duplicate removal, mutual-information feature selection, hyperparameter optimisation, soft voting, and repeated nested cross-validation.

## Recommended project structure

```text
SIT720_11.1HD/
├── SIT720_11.1HD.ipynb
├── heart.csv
├── README.md
├── requirements.txt
└── results/
```

The notebook can also load the dataset from:

```text
archive/heart.csv
```

If `archive/heart.csv` is not found, it automatically looks for `heart.csv` in the same directory as the notebook.

## Dataset

The submitted dataset is the heart disease dataset used for the reproduction experiment.

Expected shape:

```text
1025 rows x 14 columns
```

Columns:

```text
age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
target
```

The notebook audits the dataset before modelling. In the submitted data, it identifies 723 exact duplicate rows and 302 unique full rows.

## Software environment

The notebook was executed with:

```text
Python 3.12.4
pandas 2.1.4
scikit-learn 1.5.1
xgboost 2.1.3
```

It also requires NumPy, Matplotlib, and Jupyter.

A compatible installation command is:

```bash
pip install numpy pandas==2.1.4 matplotlib scikit-learn==1.5.1 xgboost==2.1.3 jupyter
```

## Running the notebook

1. Place `heart.csv` beside the notebook, or place it at `archive/heart.csv`.
2. Open a terminal in the project directory.
3. Start Jupyter:

```bash
jupyter notebook
```

4. Open `SIT720_11.1HD.ipynb`.
5. Restart the kernel.
6. Select `Run All`.
7. Allow all cells to complete before reviewing the generated outputs.

The nested cross-validation section performs repeated hyperparameter searches and may take noticeably longer than the earlier sections.

## Experimental workflow

### Part 1 - Reproduction

The notebook:

1. loads and audits the 1,025-row dataset;
2. uses an 80/20 train-test split with `random_state=42`;
3. applies standard scaling;
4. trains LR, DT, RF, XGB, Gaussian NB, and KNN;
5. evaluates a 5-fold stacking ensemble;
6. reports Accuracy, Precision, Recall, F1, and ROC-AUC;
7. compares reproduced values with the published results;
8. checks train-test duplicate overlap, random-seed sensitivity, feature importance, and inconsistencies in the paper's reported metrics.

### Part 2 - Leakage-aware evaluation

The notebook:

1. removes exact duplicate records before evaluation;
2. uses a stratified duplicate-free holdout experiment;
3. treats integer-coded categorical variables as discrete for mutual-information feature selection;
4. tunes LR, RF, XGB, and KNN with GridSearchCV using ROC-AUC;
5. combines the tuned models using soft voting;
6. compares the proposed method with duplicate-free stacking;
7. performs 5-fold outer cross-validation repeated 3 times;
8. performs model selection using 4-fold stratified inner cross-validation;
9. reports paired differences and descriptive bootstrap intervals.

The categorical variables treated as discrete during mutual-information feature selection are:

```text
sex, cp, fbs, restecg, exang, slope, ca, thal
```

The continuous variables are:

```text
age, trestbps, chol, thalach, oldpeak
```

## Hyperparameter search spaces

The Part 2 search spaces are:

```text
Logistic Regression
C: 0.1, 1.0, 10.0

Random Forest
n_estimators: 200, 500
max_depth: None, 5, 10
min_samples_leaf: 1, 2

XGBoost
n_estimators: 100, 200
max_depth: 2, 3
learning_rate: 0.03, 0.10
subsample: 0.8, 1.0

KNN
n_neighbors: 3, 5, 7, 9
weights: uniform, distance

Feature selection
k: 8, 10, or all predictors
```

## Key verification results

A successful execution should produce results close to the following saved outputs.

### Part 1

```text
Paper stacking accuracy:        0.9853
Reproduced stacking accuracy:   0.9854
Exact duplicate rows:           723
Unique full rows:               302
Test rows duplicated in train:  199 / 205 (97.1%)
```

### Part 2 repeated nested cross-validation

```text
Clean Stacking
Accuracy:   0.8289
Precision:  0.8282
Recall:     0.8700
F1:         0.8456
AUC:        0.9014

Proposed Soft Voting
Accuracy:   0.8267
Precision:  0.8100
Recall:     0.8944
F1:         0.8483
AUC:        0.9068
```

Small numerical differences may occur across software or hardware environments, although fixed random states are used where possible.

## Generated result files

Running the complete notebook creates the following CSV files:

```text
results_part1.csv
results_part2_clean_baselines.csv
results_part2_fair_comparison.csv
results_part2_context_comparison.csv
results_part2_nested_cv_folds.csv
results_part2_nested_cv_summary.csv
results_part2_nested_cv_paired.csv
results_part2_nested_cv_paired_ci.csv
results_part2_nested_cv_best_params.csv
```

These files contain the values used for the tables and analysis in the report.

## Reproducibility notes

The Part 1 reproduction follows the paper where implementation details are available and documents assumptions where the paper is underspecified.

The paper does not fully specify the original random seed, all classifier hyperparameters, or the stacking meta-classifier. The reproduction therefore uses explicit assumptions documented in the notebook and report.

Part 1 intentionally remains close to the reproduction setup. Part 2 uses stricter fold-safe pipelines and nested model selection to reduce evaluation bias.

The main methodological finding is that the near-99% Part 1 result is not maintained under duplicate-free evaluation. The Part 2 comparison is therefore interpreted as a more conservative estimate of performance on unseen record patterns rather than as an attempt to reproduce the paper's headline accuracy.

## Output interpretation

The proposed soft-voting model does not uniformly outperform clean stacking. Under repeated nested cross-validation, it produces similar overall accuracy, higher recall and AUC, and lower precision. The results are interpreted as a recall-precision trade-off rather than a universal performance improvement.

The models are research demonstrations and should not be interpreted as clinical diagnostic systems.
