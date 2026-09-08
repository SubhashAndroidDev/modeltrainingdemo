import streamlit as st
import pandas as pd
import joblib
from src.database import get_connection

# -----------------------------------------
# Page configuration
# -----------------------------------------

connection = get_connection()

cursor = connection.cursor()

cursor.execute(
    """
    INSERT INTO prediction_history (
        age,
        monthly_income,
        employment_years,
        credit_score,
        existing_emi,
        loan_amount,
        loan_term_months,
        employment_type,
        city_tier,
        prediction,
        approval_probability
    )
    VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s
    )
    """,
    (
        age,
        monthly_income,
        employment_years,
        credit_score,
        existing_emi,
        loan_amount,
        loan_term_months,
        employment_type,
        city_tier,
        int(prediction),
        float(approval_probability),
    ),
)

connection.commit()

cursor.close()
connection.close()

st.set_page_config(
    page_title="Indian Loan Prediction",
    page_icon="🏦",
    layout="centered",
)


# -----------------------------------------
# Load trained model
# -----------------------------------------

model = joblib.load(
    "models/loan_model.pkl"
)


# -----------------------------------------
# Title
# -----------------------------------------

st.title("🏦 Indian Personal Loan Prediction")

st.write(
    "Enter applicant information to estimate "
    "whether the application is likely to be "
    "approved by the trained ML model."
)


# -----------------------------------------
# User form
# -----------------------------------------

with st.form("loan_form"):

    st.subheader("Applicant Information")

    age = st.number_input(
        "Age",
        min_value=21,
        max_value=70,
        value=30,
    )

    monthly_income = st.number_input(
        "Monthly Income (₹)",
        min_value=10000,
        max_value=1000000,
        value=50000,
        step=5000,
    )

    employment_years = st.number_input(
        "Employment Experience (Years)",
        min_value=0,
        max_value=40,
        value=5,
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=700,
    )

    existing_emi = st.number_input(
        "Existing EMI (₹)",
        min_value=0,
        max_value=500000,
        value=10000,
        step=1000,
    )

    loan_amount = st.number_input(
        "Requested Loan Amount (₹)",
        min_value=50000,
        max_value=5000000,
        value=500000,
        step=50000,
    )

    loan_term_months = st.selectbox(
        "Loan Term",
        [12, 24, 36, 48, 60, 72, 84],
        index=4,
    )

    employment_type = st.selectbox(
        "Employment Type",
        ["Salaried", "Self-employed"],
    )

    city_tier = st.selectbox(
        "City Tier",
        ["Tier 1", "Tier 2", "Tier 3"],
    )

    submitted = st.form_submit_button(
        "🔮 Predict Loan Approval"
    )


# -----------------------------------------
# Prediction
# -----------------------------------------

if submitted:

    input_data = pd.DataFrame(
        {
            "age": [age],
            "monthly_income": [monthly_income],
            "employment_years": [employment_years],
            "credit_score": [credit_score],
            "existing_emi": [existing_emi],
            "loan_amount": [loan_amount],
            "loan_term_months": [loan_term_months],
            "employment_type": [employment_type],
            "city_tier": [city_tier],
        }
    )

    prediction = model.predict(
        input_data
    )[0]

    probability = model.predict_proba(
        input_data
    )[0]

    approval_probability = probability[1]


    st.divider()

    st.subheader("Prediction Result")


    if prediction == 1:

        st.success(
            "✅ Likely Loan Approval"
        )

    else:

        st.error(
            "❌ Likely Loan Rejection"
        )


    st.metric(
        "Model Approval Probability",
        f"{approval_probability:.2%}",
    )


    st.write("Applicant Data")

    st.dataframe(
        input_data,
        use_container_width=True,
    )


    st.warning(
        "This is an ML prediction for demonstration "
        "and should not be treated as a lending decision."
    )