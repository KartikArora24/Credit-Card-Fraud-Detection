
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="FraudLens AI",
    page_icon="🔍",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

# Header
st.title("🔍 FraudLens AI")

st.markdown("""
### Intelligent Credit Card Fraud Detection Dashboard
Upload transaction data and detect suspicious activities using
Machine Learning.
""")

st.divider()

# File upload
st.subheader("📂 Upload Transaction Data")

uploaded_file = st.file_uploader(
    "Upload your transaction CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    # Remove target column if present
    if "Class" in df.columns:
        df = df.drop("Class", axis=1)

    # Validate features
    expected_features = model.feature_names_in_

    missing_features = [
        feature for feature in expected_features
        if feature not in df.columns
    ]

    if missing_features:
        st.error(f"Missing features: {missing_features}")
        st.stop()

    # Arrange columns in training order
    X = df[expected_features]

    # Predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Fraud probability (assuming class 1 = fraud)
    fraud_index = list(model.classes_).index(1)
    fraud_probability = probabilities[:, fraud_index] * 100

    # Add results
    df["Prediction"] = predictions
    df["Status"] = df["Prediction"].map({
        0: "Normal",
        1: "Fraud"
    })

    df["Fraud Probability (%)"] = fraud_probability

    # Risk categorization
    def risk_level(probability):
        if probability > 80:
            return "High"
        elif probability > 40:
            return "Medium"
        else:
            return "Low"

    df["Risk Level"] = df["Fraud Probability (%)"].apply(risk_level)

    # Metrics
    total_transactions = len(df)
    fraud_count = (df["Prediction"] == 1).sum()
    fraud_percent = (
        fraud_count / total_transactions * 100
        if total_transactions > 0 else 0
    )

    st.divider()
    st.subheader("📊 Fraud Detection Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total_transactions)
    col2.metric("Fraud Detected", fraud_count)
    col3.metric("Fraud Percentage", f"{fraud_percent:.2f}%")

    # Risk distribution
    st.subheader("📈 Risk Distribution")

    risk_counts = df["Risk Level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]

    color_map = {
        "Low": "#2ECC71",
        "Medium": "#F1C40F",
        "High": "#E74C3C"
    }

    fig = px.pie(
        risk_counts,
        names="Risk Level",
        values="Count",
        hole=0.5,
        color="Risk Level",
        color_discrete_map=color_map
    )

    fig.update_layout(
        title="Transaction Risk Breakdown",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Risk metrics
    low_risk = (df["Risk Level"] == "Low").sum()
    medium_risk = (df["Risk Level"] == "Medium").sum()
    high_risk = (df["Risk Level"] == "High").sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("🟢 Low Risk", low_risk)
    col2.metric("🟡 Medium Risk", medium_risk)
    col3.metric("🔴 High Risk", high_risk)

    # Suspicious transactions
    st.divider()
    st.subheader("🚨 Suspicious Transactions")

    suspicious = df[df["Prediction"] == 1].sort_values(
        by="Fraud Probability (%)",
        ascending=False
    )

    if len(suspicious) > 0:
        st.dataframe(
            suspicious,
            use_container_width=True
        )
    else:
        st.success("No fraudulent transactions detected!")

    # Download results
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Fraud Analysis Report",
        data=csv,
        file_name="fraud_analysis.csv",
        mime="text/csv"
    )

    # Conclusion
    st.divider()
    st.subheader("📝 Analysis Summary")

    if fraud_percent < 1:
        st.success(
            "Low fraud activity detected in the uploaded dataset."
        )
    elif fraud_percent < 5:
        st.warning(
            "Elevated fraud activity detected. "
            "Further investigation is recommended."
        )
    else:
        st.error(
            "High fraud activity detected. "
            "Immediate investigation is recommended."
        )

# Sidebar
st.sidebar.title("🔍 FraudLens AI")

st.sidebar.markdown("""
### About the Project

An AI-powered dashboard for detecting
fraudulent credit card transactions.

**Model Information**
- Algorithm: Random Forest
- Features: 30
- Framework: Streamlit
- Dataset: Credit Card Transactions
""")

st.sidebar.divider()

st.sidebar.info(
    "FraudLens AI helps identify suspicious "
    "transactions through Machine Learning."
)

# Footer
st.markdown("---")

st.markdown(
    "### 🔍 FraudLens AI | Machine Learning Fraud Detection"
)

st.caption("Developed by Kartik Arora | TIET")
