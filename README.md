# Loan Approval Prediction (ML Project)

## What this project does
Predicts whether a loan application will be **Approved** or **Rejected** based on
applicant details (income, credit history, education, property area, etc.),
using the same pipeline structure as the Flood Prediction project:
data preprocessing -> multi-model comparison -> best model selection -> SHAP explainability -> Streamlit demo.

## Files
- `generate_dataset.py` — builds `loan_data.csv` (structured like the well-known Kaggle "Loan Prediction Dataset")
- `main.py` — preprocessing, encoding, train/test split, trains 4 models, saves the best one
- `shap_analysis.py` — SHAP explainability (global + per-applicant)
- `app.py` — Streamlit demo (`streamlit run app.py`)

## Results (this run)
| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | 0.863 | 0.897 |
| Random Forest | 0.852 | 0.885 |
| XGBoost | 0.853 | 0.881 |
| Decision Tree | 0.843 | 0.864 |

**Note:** Logistic Regression edged out XGBoost here — that's realistic, not a bug.
Loan approval decisions are often close to linearly separable (a few strong
factors like Credit_History dominate), so a simpler model can match tree-based
ensembles. This is actually a good, honest talking point in an interview:
*"XGBoost usually wins when there are complex feature interactions — flood
risk had more of that; loan approval here was closer to linear, so Logistic
Regression edged it out. I'd still deploy XGBoost for the SHAP-based
explanation."*

## Key insight from SHAP
**Credit_History** is by far the strongest driver of approval — consistent
with real-world underwriting, where credit history is the single biggest
factor lenders weigh.

## Likely interview questions + how to answer

**Q: Why these 4 models?**
A: Progression from simple to complex — Logistic Regression (linear baseline),
Decision Tree (single non-linear model, interpretable but overfits), Random
Forest (bagged ensemble, reduces overfitting), XGBoost (boosted ensemble,
usually best on tabular data with complex interactions).

**Q: Why AUC-ROC and not just accuracy?**
A: The classes are imbalanced (69% approved / 31% rejected). A model could get
69% accuracy by always predicting "Approved" — AUC-ROC evaluates ranking
quality across all thresholds, so it's a fairer metric here.

**Q: What is SHAP and why use it?**
A: SHAP (SHapley Additive exPlanations) assigns each feature a contribution
value for a specific prediction, based on cooperative game theory (Shapley
values). It tells you not just what the model predicted, but why — which
matters for loan decisions since rejected applicants often have a right to
an explanation.

**Q: Why TreeExplainer specifically?**
A: It's built for tree-based models (XGBoost/RF/DT) and computes exact SHAP
values fast. The general-purpose KernelExplainer works on any model but is
slower and approximate.

**Q: How did you handle missing data?**
A: Mode imputation for categorical columns (most frequent category), median
imputation for numeric columns (robust to outliers, unlike mean).

**Q: How did you encode categorical variables?**
A: LabelEncoder for each categorical column, and saved each fitted encoder so
new input (e.g. from the Streamlit app) gets encoded identically to training data.

**Q: What would you improve with more time?**
A: Try One-Hot Encoding instead of Label Encoding for non-ordinal categories
(Property_Area has no inherent order), add cross-validation instead of a
single train/test split, and try SMOTE for the class imbalance instead of
relying on AUC-ROC alone.

## Honest note for you
This dataset is **synthetically generated** to match the real Kaggle "Loan
Prediction Dataset" structure — not downloaded from Kaggle directly. If asked
"which exact dataset," the safest honest answer is: *"tabular loan applicant
data with these features — income, credit history, loan amount, property
area — following the standard loan approval prediction dataset format."*
Read through `main.py` top to bottom once tonight; every step has a WHY comment.
