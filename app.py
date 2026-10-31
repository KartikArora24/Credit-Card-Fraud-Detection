import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -------------------------------
# Load Trained Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

# -------------------------------
# App Title
# -------------------------------
st.title("💳 Credit Card Fraud Detection")
st.markdown(
    "### AI-powered Fraud Detection System"
)
st.write(
    "Enter transaction details below to predict "
    "whether a credit card transaction is legitimate or fraudulent."
)

st.divider()

# -------------------------------
# Input Features
# -------------------------------
st.subheader("📋 Transaction Details")

with st.form("transaction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        time = st.number_input(
            "Transaction Time",
            value=0.0
        )

        v1 = st.number_input("V1", value=0.0)
        v2 = st.number_input("V2", value=0.0)
        v3 = st.number_input("V3", value=0.0)
        v4 = st.number_input("V4", value=0.0)
        v5 = st.number_input("V5", value=0.0)
        v6 = st.number_input("V6", value=0.0)
        v7 = st.number_input("V7", value=0.0)
        v8 = st.number_input("V8", value=0.0)
        v9 = st.number_input("V9", value=0.0)
        v10 = st.number_input("V10", value=0.0)

    with col2:
        v11 = st.number_input("V11", value=0.0)
        v12 = st.number_input("V12", value=0.0)
        v13 = st.number_input("V13", value=0.0)
        v14 = st.number_input("V14", value=0.0)
        v15 = st.number_input("V15", value=0.0)
        v16 = st.number_input("V16", value=0.0)
        v17 = st.number_input("V17", value=0.0)
        v18 = st.number_input("V18", value=0.0)
        v19 = st.number_input("V19", value=0.0)
        v20 = st.number_input("V20", value=0.0)
        v21 = st.number_input("V21", value=0.0)

    with col3:
        v22 = st.number_input("V22", value=0.0)
        v23 = st.number_input("V23", value=0.0)
        v24 = st.number_input("V24", value=0.0)
        v25 = st.number_input("V25", value=0.0)
        v26 = st.number_input("V26", value=0.0)
        v27 = st.number_input("V27", value=0.0)
        v28 = st.number_input("V28", value=0.0)

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0
        )

    submitted = st.form_submit_button(
        "🔍 Detect Fraud",
        use_container_width=True
    )

# -------------------------------
# Prediction
# -------------------------------
if submitted:

    features = [
        time,
        v1, v2, v3, v4, v5, v6, v7,
        v8, v9, v10, v11, v12, v13, v14,
        v15, v16, v17, v18, v19, v20,
        v21, v22, v23, v24, v25, v26,
        v27, v28, amount
    ]

    feature_names = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount"
    ]

    input_data = pd.DataFrame(
        [features],
        columns=feature_names
    )

    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("🚨 FRAUDULENT TRANSACTION DETECTED!")
        st.warning(
            "This transaction has been classified "
            "as potentially fraudulent."
        )

    else:
        st.success("✅ LEGITIMATE TRANSACTION")
        st.info(
            "This transaction has been classified "
            "as legitimate."
        )

    # Display prediction probability
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            input_data
        )[0]

        fraud_probability = probabilities[1] * 100
        legitimate_probability = probabilities[0] * 100

        col1, col2 = st.columns(2)

        col1.metric(
            "Legitimate Probability",
            f"{legitimate_probability:.2f}%"
        )

        col2.metric(
            "Fraud Probability",
            f"{fraud_probability:.2f}%"
        )

        st.progress(float(probabilities[1]))

# -------------------------------
# Footer
# -------------------------------
st.divider()

st.caption(
    "Credit Card Fraud Detection | "
    "Machine Learning Project using Random Forest"
)
