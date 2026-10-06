"""
generate_dataset.py
--------------------
WHY THIS FILE EXISTS:
In your Flood Prediction project, you downloaded a ready-made Kaggle dataset.
Here, instead of pointing you to a random download link, we generate a dataset
with the SAME structure as the well-known "Loan Prediction Dataset" on Kaggle
(the one almost every ML beginner project uses: Loan_ID, Gender, Married,
ApplicantIncome, LoanAmount, Credit_History, Loan_Status, etc).

We generate it synthetically so:
1. It runs instantly, no internet/download needed.
2. The relationships between features and the target are realistic
   (Credit_History strongly affects approval, high income + low loan amount
   improves approval odds, etc) -- so your model actually learns something
   meaningful, not random noise.

If you want to swap this for the REAL Kaggle dataset later, search
"Loan Prediction Dataset" on Kaggle -- the column names here match it exactly,
so main.py would need almost no changes.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000  # number of loan applicants (similar scale to a typical Kaggle loan dataset)

# --- Categorical features ---
gender = np.random.choice(["Male", "Female"], size=N, p=[0.8, 0.2])
married = np.random.choice(["Yes", "No"], size=N, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], size=N, p=[0.55, 0.2, 0.15, 0.1])
education = np.random.choice(["Graduate", "Not Graduate"], size=N, p=[0.78, 0.22])
self_employed = np.random.choice(["Yes", "No"], size=N, p=[0.14, 0.86])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], size=N, p=[0.38, 0.38, 0.24])

# --- Numeric features ---
applicant_income = np.random.gamma(shape=5, scale=1200, size=N).round(0)
coapplicant_income = np.random.gamma(shape=2, scale=800, size=N).round(0)
coapplicant_income[np.random.rand(N) < 0.4] = 0  # 40% have no co-applicant income

loan_amount = (0.04 * applicant_income + 0.03 * coapplicant_income
               + np.random.normal(0, 30, N)).clip(10, None).round(0)
loan_amount_term = np.random.choice([360, 180, 240, 120, 60], size=N,
                                     p=[0.75, 0.1, 0.07, 0.05, 0.03])

# Credit_History is the single strongest real-world predictor in this dataset (1 = good, 0 = bad)
credit_history = np.random.choice([1, 0], size=N, p=[0.82, 0.18])

df = pd.DataFrame({
    "Loan_ID": [f"LP{1000+i}" for i in range(N)],
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
})

# --- Build Loan_Status (target) using a realistic underlying rule + noise ---
# This mimics how a real underwriting decision is influenced by multiple factors.
score = (
    2.8 * df["Credit_History"]
    + 0.0004 * df["ApplicantIncome"]
    + 0.0003 * df["CoapplicantIncome"]
    - 0.006 * df["LoanAmount"]
    + df["Education"].map({"Graduate": 0.3, "Not Graduate": -0.1})
    + df["Property_Area"].map({"Urban": 0.1, "Semiurban": 0.3, "Rural": -0.1})
    + np.random.normal(0, 0.8, N)  # noise so the model can't get 100% accuracy (realistic)
)
prob_approved = 1 / (1 + np.exp(-score))
# Use a percentile-based cutoff instead of a fixed 0.5 threshold so the class balance
# matches the real dataset (~69% approved / 31% rejected) regardless of the exact
# score formula above. This is a common trick when simulating labeled data.
cutoff = np.percentile(prob_approved, 31)
df["Loan_Status"] = np.where(prob_approved > cutoff, "Y", "N")

# Introduce a small amount of missing data -- REAL Kaggle loan datasets have missing values
# in exactly these columns, so this also mirrors the actual data-cleaning challenge you'd face.
for col, frac in [("Gender", 0.02), ("Married", 0.005), ("Dependents", 0.03),
                   ("Self_Employed", 0.05), ("LoanAmount", 0.03),
                   ("Loan_Amount_Term", 0.02), ("Credit_History", 0.08)]:
    idx = np.random.choice(N, size=int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv("loan_data.csv", index=False)
print(f"Generated loan_data.csv with {N} rows and {df.shape[1]} columns.")
print(df["Loan_Status"].value_counts(normalize=True))