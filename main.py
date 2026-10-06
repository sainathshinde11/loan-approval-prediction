"""
main.py -- Loan Default (Loan Approval) Prediction
-----------------------------------------------------
STRUCTURE MIRRORS YOUR FLOOD PREDICTION PROJECT ON PURPOSE:
    1. Load data
    2. Handle missing values + encode categoricals
    3. Train/test split
    4. Compare 4 models: Logistic Regression, Decision Tree, Random Forest, XGBoost
    5. Evaluate with Accuracy + AUC-ROC (same metrics you used before)
    6. Save the best model + the fitted encoders (needed later for SHAP + Streamlit)

READ THIS TOP TO BOTTOM ONCE -- every step has a one-line WHY comment.
That's the fastest way to be able to explain this tomorrow.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# STEP 1: Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("loan_data.csv")
df = df.drop(columns=["Loan_ID"])  # WHY: it's just an ID, carries no predictive signal

# ---------------------------------------------------------------------------
# STEP 2: Handle missing values
# WHY: Credit_History, LoanAmount, etc. have real missing values (like the actual
# Kaggle dataset does). Models can't handle NaN directly, so we fill them.
# ---------------------------------------------------------------------------
for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
    df[col] = df[col].fillna(df[col].mode()[0])  # WHY: mode = most frequent category, safe default for categorical gaps

for col in ["LoanAmount", "Loan_Amount_Term", "Credit_History"]:
    df[col] = df[col].fillna(df[col].median())  # WHY: median is robust to outliers, unlike mean

# ---------------------------------------------------------------------------
# STEP 3: Encode categorical columns to numbers
# WHY: ML models need numeric input. LabelEncoder turns "Male"/"Female" into 0/1, etc.
# We SAVE each encoder because Streamlit later needs to encode new user input
# the exact same way the training data was encoded.
# ---------------------------------------------------------------------------
categorical_cols = ["Gender", "Married", "Dependents", "Education",
                    "Self_Employed", "Property_Area"]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

target_encoder = LabelEncoder()
df["Loan_Status"] = target_encoder.fit_transform(df["Loan_Status"])  # N=0, Y=1

# ---------------------------------------------------------------------------
# STEP 4: Train/test split
# WHY: 80/20 split, stratified so both sets keep the same 69/31 approval ratio
# ---------------------------------------------------------------------------
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# STEP 5: Train and compare 4 models
# WHY these 4: same lineup as your flood project, so you can directly compare
# "simple linear model" -> "single tree" -> "ensemble of trees" -> "boosted ensemble"
# and explain in the interview WHY performance improves at each step.
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                              eval_metric="logloss", random_state=42),
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    results.append({"Model": name, "Accuracy": round(acc, 4), "AUC-ROC": round(auc, 4)})
    trained_models[name] = model
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(f"Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, preds, target_names=["Rejected", "Approved"]))

results_df = pd.DataFrame(results).sort_values("AUC-ROC", ascending=False)
print("\n\n===== MODEL COMPARISON SUMMARY =====")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# STEP 6: Save best model + encoders + test set (for SHAP script + Streamlit app)
# WHY XGBoost is typically best here (same as your flood project): it captures
# non-linear interactions (e.g. "low income BUT good credit history" patterns)
# that Logistic Regression can't, while controlling overfitting better than a
# single Decision Tree.
# ---------------------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print(f"\nBest model: {best_model_name} -> saving as loan_model.pkl")

joblib.dump(best_model, "loan_model.pkl")
joblib.dump(trained_models["XGBoost"], "loan_model_xgboost.pkl")  # kept separately for SHAP step
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
results_df.to_csv("model_comparison_results.csv", index=False)

print("\nSaved: loan_model.pkl, encoders.pkl, target_encoder.pkl, feature_columns.pkl, X_test.csv, y_test.csv")