"""
test_model_calculator.py
========================
Tests all major paths of ModelCalculator.generate_predictions.

Section  What it exercises
-------  ----------------------------------------------------------
1a       Synthetic dataset (breast cancer)
1b       Adult income dataset + pre-trained model loading
2        Basic predictions (no SHAP, no coefficients)
3        subset_results=True
4        use_coefficients=True  — feature name lists
5        use_coefficients=True, include_contributions=True — dicts
6        global_coefficients=True
7        calculate_shap=True  — feature name lists
8        calculate_shap=True, include_contributions=True — dicts
9        global_shap=True  — batched
10       global_shap=True, sample_size (direct _calculate_shap_values)
11       Conflict / ValueError guards
12       Adult income — basic predictions (all three models)
13       Adult income — global coefficients (LR)
14       Adult income — global SHAP (LR)
15       Adult income — row-wise SHAP with subset_results
"""

import os
import pickle

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import model_metrics
from model_metrics.model_calculator import ModelCalculator
from eda_toolkit import ensure_directory

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

SEP = "-" * 70


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def loadObjects(path):
    with open(path, "rb") as f:
        return pickle.load(f)


################################################################################
# Section 1a: Breast cancer dataset
################################################################################

section("1a. Breast cancer dataset")

print(f"Model Metrics version: {model_metrics.__version__}")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="outcome")

X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
    X, y, test_size=0.2, random_state=42
)
y_test_df = y_test_s.to_frame()

print(f"X_test shape : {X_test_df.shape}")
print(f"y_test shape : {y_test_df.shape}")

model1 = LogisticRegression(random_state=42, max_iter=10000).fit(X_train_df, y_train_s)
model2 = RandomForestClassifier(random_state=42).fit(X_train_df, y_train_s)
model_titles = ["Logistic Regression", "Random Forest"]

outcomes = ["outcome"]
model_dict_lr = {"model": {"outcome": model1}}
model_dict_rf = {"model": {"outcome": model2}}

print("Models trained successfully.")

################################################################################
# Section 1b: Adult income dataset
################################################################################

section("1b. Adult Income dataset")

print(f"Model Metrics version  : {model_metrics.__version__}")
print(f"Model Metrics authors  : {model_metrics.__author__}\n")

# Anchor paths to the script's own location so they resolve correctly
# regardless of which directory you run python from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

model_path = os.path.join(PROJECT_ROOT, "model_files", "results")
data_path = os.path.join(PROJECT_ROOT, "model_files")
image_path_png = os.path.join(data_path, "images", "png_images")
image_path_svg = os.path.join(data_path, "images", "svg_images")

model_lr = loadObjects(os.path.join(model_path, "LogisticRegression.pkl"))
model_dt = loadObjects(os.path.join(model_path, "DecisionTreeClassifier.pkl"))
model_rf_ai = loadObjects(os.path.join(model_path, "RandomForestClassifier.pkl"))

X_test_ai = pd.read_parquet(os.path.join(data_path, "X_test.parquet"))
y_test_ai = pd.read_parquet(os.path.join(data_path, "y_test.parquet"))

print(f"X_test_ai shape   : {X_test_ai.shape}")
print(f"y_test_ai shape   : {y_test_ai.shape}")
print(f"y_test_ai columns : {list(y_test_ai.columns)}")

# Inspect model wrapper structure
print("\nModel wrapper inspection:")
for label, m in [("LR", model_lr), ("DT", model_dt), ("RF", model_rf_ai)]:
    print(f"  {label} keys: {list(m.keys())}")
    for k, v in m.items():
        print(
            f"    [{k}]: {type(v)}"
            f"  has predict={hasattr(v, 'predict')}"
            f"  has predict_proba={hasattr(v, 'predict_proba')}"
        )

AI_OUTCOME = y_test_ai.columns[0]
model_dict_ai_lr = {"model": {AI_OUTCOME: model_lr}}
model_dict_ai_rf = {"model": {AI_OUTCOME: model_rf_ai}}
outcomes_ai = [AI_OUTCOME]
print(f"\nAdult income outcome: '{AI_OUTCOME}'")

################################################################################
# Section 2: Basic predictions
################################################################################

section("2. Basic predictions — no SHAP, no coefficients")

calc_lr = ModelCalculator(model_dict=model_dict_lr, outcomes=outcomes, top_n=3)

results_basic = calc_lr.generate_predictions(
    X_test=X_test_df,
    y_test=y_test_df,
)

print(f"Shape: {results_basic.shape}")
print(results_basic.head().to_string())

row_check = results_basic[["TP", "FN", "FP", "TN"]].sum(axis=1)
assert (row_check == 1).all(), "Each row should belong to exactly one quadrant"
print(f"\nTP/FN/FP/TN sanity check  : {PASS}")
print("\nConfusion matrix summary:")
print(results_basic[["TP", "FN", "FP", "TN"]].sum().to_string())

################################################################################
# Section 3: subset_results=True
################################################################################

section("3. subset_results=True")

results_subset = calc_lr.generate_predictions(
    X_test=X_test_df,
    y_test=y_test_df,
    subset_results=True,
)

print(f"Columns returned: {list(results_subset.columns)}")
assert set(results_subset.columns) == {"TP", "FN", "FP", "TN", "y_pred_proba"}
print(f"Subset columns assertion  : {PASS}")
print(results_subset.head().to_string())

################################################################################
# Section 4: use_coefficients=True, feature name lists
################################################################################

section("4. use_coefficients=True — feature name lists")

results_coef = calc_lr.generate_predictions(
    X_test=X_test_df,
    y_test=y_test_df,
    use_coefficients=True,
)

coef_col = f"top_{calc_lr.top_n}_coefficients"
sample = results_coef[coef_col].iloc[0]
print(f"Coefficient column : '{coef_col}'")
print(f"Sample row value   : {sample}")
assert isinstance(sample, list), "Expected a list of feature names"
assert (
    len(sample) == calc_lr.top_n
), f"Expected {calc_lr.top_n} features, got {len(sample)}"
print(f"Coefficient list assertion : {PASS}")

################################################################################
# Section 5: use_coefficients=True, include_contributions=True
################################################################################

section("5. use_coefficients=True, include_contributions=True — contribution dicts")

results_coef_contrib = calc_lr.generate_predictions(
    X_test=X_test_df,
    y_test=y_test_df,
    use_coefficients=True,
    include_contributions=True,
)

sample_contrib = results_coef_contrib[coef_col].iloc[0]
print(f"Sample row value : {sample_contrib}")
assert isinstance(sample_contrib, dict), "Expected a dict of {feature: contribution}"
assert len(sample_contrib) == calc_lr.top_n
print(f"Contribution dict assertion : {PASS}")

################################################################################
# Section 6: global_coefficients=True
################################################################################

section("6. global_coefficients=True")

global_coef_df = calc_lr.generate_predictions(
    X_test=X_test_df,
    y_test=y_test_df,
    global_coefficients=True,
)

print(f"Type: {type(global_coef_df)}")
assert isinstance(global_coef_df, pd.DataFrame)
assert "Feature" in global_coef_df.columns
assert "Coefficient" in global_coef_df.columns
print(f"Global coefficients assertion : {PASS}")
print(global_coef_df.to_string())

################################################################################
# Section 7: calculate_shap=True, feature name lists
################################################################################

section("7. calculate_shap=True — feature name lists")

results_shap = calc_lr.generate_predictions(
    X_test=X_test_df,
    y_test=y_test_df,
    calculate_shap=True,
)

shap_col = f"top_{calc_lr.top_n}_features"
sample_shap = results_shap[shap_col].iloc[0]
print(f"SHAP column    : '{shap_col}'")
print(f"Sample row value : {sample_shap}")
assert isinstance(sample_shap, list)
assert len(sample_shap) == calc_lr.top_n
print(f"SHAP feature list assertion : {PASS}")

################################################################################
# Section 8: calculate_shap=True, include_contributions=True
################################################################################

section("8. calculate_shap=True, include_contributions=True — contribution dicts")

results_shap_contrib = calc_lr.generate_predictions(
    X_test=X_test_df,
    y_test=y_test_df,
    calculate_shap=True,
    include_contributions=True,
)

sample_shap_contrib = results_shap_contrib[shap_col].iloc[0]
print(f"Sample row value : {sample_shap_contrib}")
assert isinstance(sample_shap_contrib, dict)
assert len(sample_shap_contrib) == calc_lr.top_n
print(f"SHAP contribution dict assertion : {PASS}")

################################################################################
# Section 9: global_shap=True
################################################################################

section("9. global_shap=True — batched global SHAP")

global_shap_df = calc_lr.generate_predictions(
    X_test=X_test_df,
    y_test=y_test_df,
    global_shap=True,
)

print(f"Type: {type(global_shap_df)}")
assert isinstance(global_shap_df, pd.DataFrame)
assert "Feature" in global_shap_df.columns
assert "SHAP Value" in global_shap_df.columns
assert len(global_shap_df) == X_test_df.shape[1], "Should have one row per feature"
print(f"Global SHAP assertion : {PASS}")
print(global_shap_df.to_string())

################################################################################
# Section 10: global_shap with sample_size (direct call)
################################################################################

section("10. global_shap=True with sample_size — direct _calculate_shap_values call")

global_shap_sampled = calc_lr._calculate_shap_values(
    model=model1,
    X_test_m=X_test_df,
    global_shap=True,
    sample_size=50,
)

assert isinstance(global_shap_sampled, pd.DataFrame)
assert "Feature" in global_shap_sampled.columns
print(f"global_shap with sample_size assertion : {PASS}")
print(global_shap_sampled.to_string())

################################################################################
# Section 11: Conflict / ValueError guards
################################################################################

section("11. Conflict / ValueError guards")

conflict_cases = [
    dict(calculate_shap=True, use_coefficients=True),
    dict(global_shap=True, global_coefficients=True),
    dict(global_coefficients=True, subset_results=True),
    dict(global_shap=True, subset_results=True),
]

for kwargs in conflict_cases:
    try:
        calc_lr.generate_predictions(
            X_test=X_test_df,
            y_test=y_test_df,
            **kwargs,
        )
        print(f"{FAIL} — no error raised for {list(kwargs.keys())}")
    except ValueError as e:
        print(f"{PASS} — ValueError for {list(kwargs.keys())}: {e}")

################################################################################
# Section 12: Adult income: basic predictions (all three models)
################################################################################

section("12. Adult income: basic predictions (all three models)")

for label, md in [
    ("Logistic Regression", {"model": {AI_OUTCOME: model_lr}}),
    ("Decision Tree", {"model": {AI_OUTCOME: model_dt}}),
    ("Random Forest", {"model": {AI_OUTCOME: model_rf_ai}}),
]:
    calc = ModelCalculator(model_dict=md, outcomes=outcomes_ai, top_n=3)
    res = calc.generate_predictions(X_test=X_test_ai, y_test=y_test_ai)
    tp = res["TP"].sum()
    fn = res["FN"].sum()
    fp = res["FP"].sum()
    tn = res["TN"].sum()
    print(
        f"{label:25s}  TP={tp:5d}  FN={fn:5d}  FP={fp:5d}  TN={tn:5d}"
        f"  shape={res.shape}"
    )

################################################################################
# Section 13: Adult income, global coefficients (LR)
################################################################################

section("13. Adult income — global coefficients (Logistic Regression)")

calc_ai_lr = ModelCalculator(
    model_dict={"model": {AI_OUTCOME: model_lr}},
    outcomes=outcomes_ai,
    top_n=5,
)

global_coef_ai = calc_ai_lr.generate_predictions(
    X_test=X_test_ai,
    y_test=y_test_ai,
    global_coefficients=True,
)

assert "Coefficient" in global_coef_ai.columns
print(f"Global coefficients assertion : {PASS}")
print("\nTop 10 features by absolute coefficient:")
print(global_coef_ai.head(10).to_string())

################################################################################
# Section 14: Adult income: global SHAP (LR)
################################################################################

section("14. Adult income: global SHAP (Logistic Regression)")

global_shap_ai = calc_ai_lr.generate_predictions(
    X_test=X_test_ai,
    y_test=y_test_ai,
    global_shap=True,
)

assert "SHAP Value" in global_shap_ai.columns
print(f"Global SHAP assertion : {PASS}")
print("\nTop 10 features by absolute SHAP value:")
print(global_shap_ai.head(10).to_string())

# Optional sampled version
print("\n--- Sampled version (200 rows) ---")
global_shap_ai_sampled = calc_ai_lr._calculate_shap_values(
    model=model_lr,
    X_test_m=X_test_ai.select_dtypes(include=["number"]),
    global_shap=True,
    sample_size=200,
)
print(global_shap_ai_sampled.head(10).to_string())

################################################################################
# Section 15: Adult income: row-wise SHAP with subset_results
################################################################################

section("15. Adult income: row-wise SHAP with subset_results")

results_ai_shap_subset = calc_ai_lr.generate_predictions(
    X_test=X_test_ai,
    y_test=y_test_ai,
    calculate_shap=True,
    subset_results=True,
)

print(f"Columns : {list(results_ai_shap_subset.columns)}")
print(f"Shape   : {results_ai_shap_subset.shape}")
print(results_ai_shap_subset.head().to_string())

print(f"\n{SEP}")
print("  All sections complete.")
print(SEP)
