# MSc Business Analytics Project - Reproducible Analysis

**Project title:** Predicting E-Commerce Purchase Completion from Online Session Behaviour: A Comparative Machine Learning Analysis Using Python

## Dataset
Sakar, C. and Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. DOI: 10.24432/C5F88Q.

The exact analysis file is `online_shoppers_intention.csv` and contains 12,330 sessions, 17 predictors and the binary `Revenue` target.

## Reproduce the analysis
Place `Python_Analysis_FINAL.py` and `online_shoppers_intention.csv` in the same folder. Install the packages listed in `requirements.txt`, then run:

`python Python_Analysis_FINAL.py`

The script reproduces data-quality checks, descriptive analysis, correlations, Mann-Whitney U tests, chi-square tests, logistic regression, decision tree, random forest, held-out evaluation, PR-AUC, five-fold stratified cross-validation, feature importance, permutation importance and the sensitivity analysis excluding `PageValues`. The full five-fold random-forest validation uses 500 trees per fit and may take several minutes depending on the computer.

## Key verified results
Random Forest held-out accuracy = 0.897810; F1 = 0.594855; ROC-AUC = 0.918458; PR-AUC = 0.730169.

Random Forest five-fold mean ROC-AUC = 0.928179; mean PR-AUC = 0.746005.

Random Forest without `PageValues`: ROC-AUC = 0.760603; PR-AUC = 0.356611.

## Repository submission
Upload this folder to the student's own GitHub or GitLab repository and add the repository URL to Appendix D of the dissertation before submission.


## Open the Jupyter Notebook
1. Keep `Python_Analysis_Reproducible.ipynb` and `online_shoppers_intention.csv` in the same folder.
2. Open Jupyter Notebook or JupyterLab.
3. Open `Python_Analysis_Reproducible.ipynb`.
4. Choose a Python 3 kernel and run the cells from top to bottom.

The notebook and Python script use only relative paths, so no local computer-specific file paths are required.

## Main analytical methods
The repository documents data-quality checks, descriptive statistics, Pearson/point-biserial correlation, Mann-Whitney U tests with rank-biserial effect sizes, chi-square tests with Cramer's V, Logistic Regression, Decision Tree, Random Forest, held-out evaluation, ROC-AUC, PR-AUC, five-fold stratified cross-validation, majority-class baseline comparison, Random Forest impurity and permutation importance, and a sensitivity analysis excluding `PageValues`.

## Important note on PageValues
`PageValues` is retained in the primary model because it is part of the published dataset, but a second analysis excludes it to assess dependence on this analytics-derived feature and its availability at a real-time prediction point.
