"""
shap_analysis.py
------------------
WHY: In your Flood Prediction project you used SHAP to explain WHY the model
predicted flood risk for a given input, not just WHAT it predicted. Same idea
here: SHAP tells us which features push a loan application toward "Approved"
or "Rejected", and by how much.

We run SHAP on the XGBoost model specifically (not Logistic Regression) because
SHAP's TreeExplainer is fast and exact for tree-based models -- same reason
your flood project used TreeExplainer on XGBoost.

INTERVIEW TALKING POINT:
"SHAP assigns each feature a contribution value for a specific prediction --
positive values push toward Approved, negative push toward Rejected. This
matters for loan systems specifically because regulators often require lenders
to explain rejections (right to explanation), not just output a black-box score."
"""

import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # WHY: no display available in this environment, save plots to file instead
import matplotlib.pyplot as plt

model = joblib.load("loan_model_xgboost.pkl")
X_test = pd.read_csv("X_test.csv")

# WHY TreeExplainer: it's built specifically for tree-based models (XGBoost, RF,
# Decision Trees) and computes EXACT Shapley values fast, unlike the general-purpose
# KernelExplainer which is slower and approximate.
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# --- Global importance: which features matter most ACROSS all applicants ---
plt.figure()
shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
plt.title("Global Feature Importance (mean |SHAP value|)")
plt.tight_layout()
plt.savefig("shap_summary_bar.png", dpi=120)
plt.close()

# --- Beeswarm: shows both importance AND direction (does high value push approve/reject) ---
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary_beeswarm.png", dpi=120)
plt.close()

# --- Local explanation: WHY was ONE specific applicant approved/rejected ---
# WHY THIS MATTERS: this is the "explain a single decision" use case -- the same
# thing you'd say in an interview: "for applicant #5, credit history contributed
# +0.8 toward approval while a high loan amount contributed -0.3."
sample_idx = 5
sample = X_test.iloc[[sample_idx]]
sample_shap = explainer.shap_values(sample)

print(f"\n--- Local explanation for applicant #{sample_idx} ---")
print(sample.T)
contributions = pd.Series(sample_shap[0], index=X_test.columns).sort_values(key=abs, ascending=False)
print("\nTop feature contributions (positive = pushes toward Approved):")
print(contributions)

print("\nSaved: shap_summary_bar.png, shap_summary_beeswarm.png")