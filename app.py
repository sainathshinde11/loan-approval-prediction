"""
app.py -- Streamlit demo for Loan Approval Prediction
--------------------------------------------------------
WHY: your Flood Prediction project had a Streamlit demo so a non-technical
person (or an interviewer) could enter inputs and see a live prediction.
Same pattern here: take applicant details as input, encode them the SAME way
main.py encoded training data (using the SAVED encoders), predict, and show
the SHAP-based explanation for that specific prediction.

RUN THIS WITH: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

# --- Load everything saved by main.py ---
# WHY: we must use the SAME model, encoders, and column order used in training,
# otherwise predictions on new input would be meaningless.
model = joblib.load("loan_model.pkl")
xgb_model = joblib.load("loan_model_xgboost.pkl")  # used only for the SHAP explanation
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🏦 Loan Approval Prediction")
st.caption("ML pipeline: preprocessing -> LR/DT/RF/XGBoost comparison -> SHAP explainability")

st.subheader("Applicant Details")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("Applicant Income (monthly)", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income (monthly)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, value=150)
    loan_term = st.selectbox("Loan Term (days)", [360, 180, 240, 120, 60])
    credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])

if st.button("Predict Loan Approval", type="primary"):
    # --- Build a single-row DataFrame matching training data's raw column format ---
    raw_input = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": 1 if credit_history.startswith("Good") else 0,
        "Property_Area": property_area,
    }])

    # --- Encode categoricals using the SAME encoders fit during training ---
    # WHY: this is the step people most often get wrong -- if you fit a NEW
    # LabelEncoder here, "Male" might become 1 instead of 0, silently breaking
    # the model. Reusing the saved encoder guarantees consistency.
    encoded_input = raw_input.copy()
    for col, le in encoders.items():
        encoded_input[col] = le.transform(encoded_input[col])

    encoded_input = encoded_input[feature_columns]  # ensure exact column order

    pred = model.predict(encoded_input)[0]
    prob = model.predict_proba(encoded_input)[0][1]
    label = target_encoder.inverse_transform([pred])[0]

    if label == "Y":
        st.success(f"✅ Loan likely APPROVED (confidence: {prob:.1%})")
    else:
        st.error(f"❌ Loan likely REJECTED (confidence: {1-prob:.1%})")

    # --- SHAP explanation for THIS specific applicant ---
    st.subheader("Why this prediction?")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(encoded_input)

    contributions = pd.Series(shap_values[0], index=feature_columns).sort_values(key=abs, ascending=False)
    st.write("Feature contributions (positive pushes toward Approved, negative toward Rejected):")
    st.dataframe(contributions.rename("SHAP value").to_frame())

    fig, ax = plt.subplots()
    colors = ["green" if v > 0 else "red" for v in contributions.values]
    ax.barh(contributions.index[::-1], contributions.values[::-1], color=colors[::-1])
    ax.set_xlabel("SHAP value (impact on prediction)")
    st.pyplot(fig)

st.divider()
st.caption("Model comparison results (from main.py):")
st.dataframe(pd.read_csv("model_comparison_results.csv"))