"""Reproducible analysis for the MSc Business Analytics dissertation.

Project: Predicting E-Commerce Purchase Completion from Online Session Behaviour: A Comparative Machine Learning Analysis Using Python

Dataset: Sakar, C. and Kastro, Y. (2018), UCI Machine Learning Repository,
DOI 10.24432/C5F88Q.
"""
from pathlib import Path
import warnings, math
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve)
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
DATA_FILE = Path('online_shoppers_intention.csv')
OUT = Path('analysis_outputs')
OUT.mkdir(exist_ok=True)

# 1. Load and quality-check the exact dataset used in the dissertation.
df = pd.read_csv(DATA_FILE)
for col in ['Weekend', 'Revenue']:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.upper().map({'TRUE': True, 'FALSE': False})
print('Dataset shape:', df.shape)
print('Missing values:', int(df.isna().sum().sum()))
print('Exact repeated profiles:', int(df.duplicated().sum()))
print(df['Revenue'].value_counts())

# 2. Descriptive analysis.
revenue_table = pd.DataFrame({
    'Count': df['Revenue'].value_counts().reindex([False, True]),
    'Percentage': df['Revenue'].value_counts(normalize=True).reindex([False, True]) * 100
})
revenue_table.index = ['No purchase', 'Purchase']
print('\nRevenue distribution\n', revenue_table)

key_vars = ['ProductRelated', 'ProductRelated_Duration', 'BounceRates',
            'ExitRates', 'PageValues', 'SpecialDay']
print('\nBehavioural group means\n', df.groupby('Revenue')[key_vars].mean().T)

# 3. Association analysis.
corr_df = df.copy()
corr_df['Revenue_binary'] = corr_df['Revenue'].astype(int)
correlations = corr_df.select_dtypes(include=np.number).corr()['Revenue_binary'] \
    .drop('Revenue_binary').sort_values(key=lambda s: s.abs(), ascending=False)
print('\nCorrelations with Revenue\n', correlations)

numeric_tests = []
for var in key_vars:
    no_purchase = df.loc[~df['Revenue'], var].values
    purchase = df.loc[df['Revenue'], var].values
    u_no, p = mannwhitneyu(no_purchase, purchase, alternative='two-sided')
    u_purchase = len(no_purchase) * len(purchase) - u_no
    u_reported = min(u_no, u_purchase)
    rbc = 2 * u_purchase / (len(no_purchase) * len(purchase)) - 1
    numeric_tests.append([var, u_reported, p, rbc])
numeric_tests = pd.DataFrame(numeric_tests,
    columns=['Variable', 'Mann_Whitney_U', 'p_value', 'Rank_biserial_effect'])
print('\nMann-Whitney U tests\n', numeric_tests)

categorical_tests = []
for var in ['VisitorType', 'Month', 'Weekend']:
    table = pd.crosstab(df[var], df['Revenue'])
    chi2, p, dof, _ = chi2_contingency(table)
    n = table.values.sum()
    cramer_v = math.sqrt(chi2 / (n * min(table.shape[0]-1, table.shape[1]-1)))
    categorical_tests.append([var, chi2, dof, p, cramer_v])
categorical_tests = pd.DataFrame(categorical_tests,
    columns=['Variable', 'Chi_square', 'df', 'p_value', 'Cramers_V'])
print('\nCategorical association tests\n', categorical_tests)

# 4. Preprocessing and train-test split.
X = df.drop(columns='Revenue').copy()
y = df['Revenue'].astype(int)
categorical_cols = ['Month', 'OperatingSystems', 'Browser', 'Region',
                    'TrafficType', 'VisitorType', 'Weekend']
numeric_cols = [c for c in X.columns if c not in categorical_cols]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

pre_scaled = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])
pre_unscaled = ColumnTransformer([
    ('num', 'passthrough', numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])

models = {
    'Logistic Regression': Pipeline([
        ('preprocess', pre_scaled),
        ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
    'Decision Tree': Pipeline([
        ('preprocess', pre_unscaled),
        ('model', DecisionTreeClassifier(random_state=RANDOM_STATE, min_samples_leaf=5))]),
    'Random Forest': Pipeline([
        ('preprocess', pre_unscaled),
        ('model', RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE,
                                         min_samples_leaf=2, n_jobs=-1))])
}

# 5. Held-out test performance.
results, predictions, probabilities = [], {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    predictions[name], probabilities[name] = pred, prob
    results.append([name, accuracy_score(y_test, pred),
                    precision_score(y_test, pred, zero_division=0),
                    recall_score(y_test, pred, zero_division=0),
                    f1_score(y_test, pred, zero_division=0),
                    roc_auc_score(y_test, prob),
                    average_precision_score(y_test, prob)])
results = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall',
                                        'F1_score', 'ROC_AUC', 'PR_AUC'])
print('\nHeld-out model comparison\n', results)

# Majority-class baseline for the imbalanced outcome.
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_prob = dummy.predict_proba(X_test)[:, 1]
print('\nMajority baseline accuracy:', accuracy_score(y_test, dummy_pred))
print('Majority baseline PR-AUC:', average_precision_score(y_test, dummy_prob))

# 6. Five-fold stratified cross-validation.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {'Accuracy': 'accuracy', 'Precision': 'precision', 'Recall': 'recall',
           'F1_score': 'f1', 'ROC_AUC': 'roc_auc', 'PR_AUC': 'average_precision'}
cv_rows = []
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    row = {'Model': name}
    for metric in scoring:
        values = scores['test_' + metric]
        row[metric + '_mean'] = values.mean()
        row[metric + '_SD'] = values.std(ddof=1)
    cv_rows.append(row)
cv_results = pd.DataFrame(cv_rows)
print('\nFive-fold cross-validation\n', cv_results)

# 7. Feature importance.
rf = models['Random Forest']
pre = rf.named_steps['preprocess']
rf_model = rf.named_steps['model']
feature_names = pre.get_feature_names_out()
importance = pd.DataFrame({'Feature': feature_names,
                           'Importance': rf_model.feature_importances_}) \
                .sort_values('Importance', ascending=False)
importance['Feature'] = importance['Feature'].str.replace(r'^num__|^cat__', '', regex=True)
print('\nRandom-forest feature importance\n', importance.head(15))

perm = permutation_importance(rf, X_test, y_test, n_repeats=5, scoring='roc_auc',
                              random_state=RANDOM_STATE, n_jobs=-1, max_samples=0.75)
perm_importance = pd.DataFrame({'Feature': X_test.columns,
                                'Mean_ROC_AUC_Decrease': perm.importances_mean,
                                'SD': perm.importances_std}) \
                    .sort_values('Mean_ROC_AUC_Decrease', ascending=False)
print('\nPermutation importance\n', perm_importance.head(15))

# 8. Sensitivity analysis without PageValues.
X_reduced = df.drop(columns=['Revenue', 'PageValues']).copy()
num_reduced = [c for c in X_reduced.columns if c not in categorical_cols]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reduced, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
pre_r_scaled = ColumnTransformer([
    ('num', StandardScaler(), num_reduced),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])
pre_r_unscaled = ColumnTransformer([
    ('num', 'passthrough', num_reduced),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])
reduced_models = {
    'Logistic Regression': Pipeline([('preprocess', pre_r_scaled),
        ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
    'Decision Tree': Pipeline([('preprocess', pre_r_unscaled),
        ('model', DecisionTreeClassifier(random_state=RANDOM_STATE, min_samples_leaf=5))]),
    'Random Forest': Pipeline([('preprocess', pre_r_unscaled),
        ('model', RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE,
                                         min_samples_leaf=2, n_jobs=-1))])}
reduced_results = []
for name, model in reduced_models.items():
    model.fit(Xr_train, yr_train)
    pred = model.predict(Xr_test)
    prob = model.predict_proba(Xr_test)[:, 1]
    reduced_results.append([name, accuracy_score(yr_test, pred),
                            precision_score(yr_test, pred, zero_division=0),
                            recall_score(yr_test, pred, zero_division=0),
                            f1_score(yr_test, pred, zero_division=0),
                            roc_auc_score(yr_test, prob),
                            average_precision_score(yr_test, prob)])
reduced_results = pd.DataFrame(reduced_results, columns=results.columns)
print('\nSensitivity analysis without PageValues\n', reduced_results)

# 9. Save machine-readable outputs.
results.to_csv(OUT/'heldout_model_comparison.csv', index=False)
cv_results.to_csv(OUT/'cross_validation_5fold.csv', index=False)
numeric_tests.to_csv(OUT/'significance_numeric.csv', index=False)
categorical_tests.to_csv(OUT/'significance_categorical.csv', index=False)
importance.to_csv(OUT/'rf_impurity_feature_importance.csv', index=False)
perm_importance.to_csv(OUT/'rf_permutation_importance.csv', index=False)
reduced_results.to_csv(OUT/'sensitivity_without_pagevalues.csv', index=False)
for name in models:
    pd.DataFrame(confusion_matrix(y_test, predictions[name]),
                 index=['Actual_NoPurchase', 'Actual_Purchase'],
                 columns=['Pred_NoPurchase', 'Pred_Purchase']) \
      .to_csv(OUT/f"confusion_{name.lower().replace(' ', '_')}.csv")

# 10. Core figures.
fig, ax = plt.subplots(figsize=(7, 5))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probabilities[name]):.4f})")
ax.plot([0, 1], [0, 1], linestyle='--', label='Chance')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves for Purchase Prediction Models')
ax.legend()
plt.tight_layout()
plt.savefig(OUT/'roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print('\nAnalysis complete. Outputs saved in:', OUT.resolve())
