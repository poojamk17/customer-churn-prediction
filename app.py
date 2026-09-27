
import streamlit as st
import pandas as pd
from predictor import ChurnPredictor
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return ChurnPredictor("churn_model.json")

predictor = load_model()

# -----------------------------
# Header
# -----------------------------
st.title("📊 Customer Churn Prediction")
st.markdown(
    "Predict customer churn risk using a trained "
    "Random Forest machine learning model."
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("About This Project")

st.sidebar.info(
    """
    **Customer Churn Prediction**

    Machine Learning: Random Forest

    Dataset: IBM Telco Customer Churn

    Classification threshold: 0.30

    Purpose: Identify customers who may be
    at risk of leaving the service.
    """
)


# -----------------------------
# Model Performance Dashboard
# -----------------------------
st.header("📈 Model Performance Dashboard")

st.markdown(
    "Performance of the Random Forest model on the "
    "held-out test dataset of 1,409 customers."
)

# Model metrics measured on the test set
metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Accuracy", "76.30%")

with metric2:
    st.metric("Precision", "53.69%")

with metric3:
    st.metric("Recall", "77.81%")

with metric4:
    st.metric("ROC-AUC", "0.8423")

st.caption(
    "These metrics use the original 0.50 classification "
    "threshold. The app uses a cost-based threshold of 0.30 "
    "for customer predictions."
)

st.divider()

# Confusion matrix at selected threshold
st.subheader("Confusion Matrix — Threshold 0.30")

st.markdown(
    "The matrix below shows the model's test-set results "
    "at the selected threshold of 0.30."
)

# Actual results from your cost analysis
tn, fp, fn, tp = 598, 437, 38, 336

matrix_col1, matrix_col2 = st.columns([3, 2])

with matrix_col1:
    st.markdown("**Actual vs. Predicted**")

    confusion_df = pd.DataFrame(
        [
            [tn, fp],
            [fn, tp]
        ],
        index=["Actual: Stayed", "Actual: Churned"],
        columns=["Predicted: Stayed", "Predicted: Churned"]
    )

    st.dataframe(
        confusion_df,
        use_container_width=True
    )

with matrix_col2:
    st.markdown("**Interpretation**")

    st.success(f"True negatives: {tn}")
    st.error(f"False positives: {fp}")
    st.error(f"False negatives: {fn}")
    st.success(f"True positives: {tp}")

# Threshold-specific metrics
threshold_recall = tp / (tp + fn)
threshold_precision = tp / (tp + fp)
threshold_accuracy = (tp + tn) / (tp + tn + fp + fn)

st.markdown("### Metrics at Threshold 0.30")

m1, m2, m3 = st.columns(3)

m1.metric("Recall", f"{threshold_recall:.2%}")
m2.metric("Precision", f"{threshold_precision:.2%}")
m3.metric("Accuracy", f"{threshold_accuracy:.2%}")

st.divider()

# Cost analysis
st.subheader("Business Cost Analysis")

cost1, cost2, cost3 = st.columns(3)

with cost1:
    st.metric("False Negative Cost", "110")

with cost2:
    st.metric("False Positive Cost", "10")

with cost3:
    st.metric("Total Test-Set Cost", "8,550")

st.info(
    "At threshold 0.30, the model identifies 336 of 374 "
    "actual churners and flags 773 customers for retention "
    "outreach. The estimated cost uses assumed penalties "
    "of 110 per missed churner and 10 per unnecessary contact."
)

# -----------------------------
# Customer Input Form
# -----------------------------
st.subheader("Enter Customer Details")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Has Partner?",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Has Dependents?",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

    with col2:
        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    # Additional billing details
    st.markdown("### Billing Information")

    col4, col5, col6 = st.columns(3)

    with col4:
        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with col5:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col6:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=monthly_charges * tenure,
        step=10.0
    )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    result = predictor.predict(customer)

    probability = result["probability"]
    status = result["status"]

    st.divider()
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )

        st.progress(probability)

    with col2:
        if result["prediction"] == 1:
            st.error(f"⚠️ {status}")
        else:
            st.success(f"✅ {status}")

    if result["prediction"] == 1:
        st.warning(
            "This customer is above the model's "
            "30% churn-risk threshold. Consider "
            "reviewing retention options."
        )
    else:
        st.info(
            "This customer is below the model's "
            "30% churn-risk threshold."
        )

    st.caption(
        "This is a model prediction, not a guarantee "
        "of future customer behavior."
    )