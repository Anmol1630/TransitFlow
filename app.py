import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load model and initialize encoders/scalers
label_encoder = LabelEncoder()
scaler = StandardScaler()
model = pickle.load(open('7_logistic_model.pkl', 'rb'))  # Fixed filename spacing
df = pd.read_csv("7_churn.csv")  # Fixed filename spacing

# Streamlit App Title
st.title("💡 Customer Churn Prediction using Logistic Regression")
st.markdown("Predict whether a customer is likely to **Churn or Stay** based on their information.")

# --- INPUT SECTION ---
st.header("📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Select Gender", options=['Female', 'Male'])
    SeniorCitizen = st.selectbox("Are you a Senior Citizen?", options=['Yes', 'No'])
    Partner = st.selectbox("Do you have a Partner?", options=['Yes', 'No'])
    Dependents = st.selectbox("Are you a Dependent?", options=['Yes', 'No'])
    tenure = st.number_input("Enter Tenure (in months)", min_value=0, max_value=100, value=12)

with col2:
    PhoneService = st.selectbox("Do you have Phone Service?", options=['Yes', 'No'])
    MultipleLines = st.selectbox("Do you have Multiple Lines?", options=['Yes', 'No', 'No phone service'])
    Contract = st.selectbox("Type of Contract", options=['Month-to-month', 'One year', 'Two year'])
    TotalCharges = st.number_input("Enter Total Charges", min_value=0.0, value=1000.0)

# --- PREDICTION FUNCTION ---
def prediction(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, Contract, TotalCharges):
    # Prepare input data
    data = {
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'Contract': [Contract],
        'TotalCharges': [TotalCharges]
    }
    df = pd.DataFrame(data)

    # Encode categorical columns (⚠️ Normally, you'd load trained encoders)
    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'Contract']
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Scale numerical values (⚠️ Normally, you'd use the scaler fitted on training data)
    df = scaler.fit_transform(df)

    # Predict churn
    result = model.predict(df).reshape(1, -1)
    return result[0]

# --- TIPS SECTION ---
churn_tips = [
    "Identify the Reasons: Conduct surveys to find out why customers leave.",
    "Improve Communication: Keep customers informed and valued.",
    "Enhance Experience: Upgrade services and provide great support.",
    "Offer Incentives: Loyalty programs and exclusive discounts.",
    "Personalize Offers: Use data to tailor deals and communication.",
    "Monitor Engagement: Track customer activity and satisfaction.",
    "Use Predictive Analytics: Detect early signs of churn.",
    "Create Feedback Loops: Use feedback to improve continuously.",
    "Train Support Teams: Equip them to handle issues effectively.",
    "Study Competitors: Stay ahead with better offerings."
]

retention_tips = [
    "Provide Excellent Support and Fast Resolutions.",
    "Launch Customer Loyalty or Rewards Programs.",
    "Communicate Regularly with Value-driven Content.",
    "Ensure Product/Service Quality Consistency.",
    "Resolve Issues Promptly and Professionally.",
    "Build Trust through Transparency.",
    "Offer Value-added Benefits.",
    "Simplify Onboarding and Renewals.",
    "Stay Responsive Across Platforms.",
    "Show Appreciation to Loyal Customers."
]

# --- PREDICT BUTTON ---
if st.button("🔮 Predict Churn"):
    result = prediction(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, Contract, TotalCharges)
    if result == 1:
        st.error("🚨 The customer is **likely to churn!**")
        st.markdown("### 🧭 10 Tips to Reduce Churn")
        st.write(pd.DataFrame({"Churn Prevention Tips": churn_tips}))
    else:
        st.success("✅ The customer is **not likely to churn!**")
        st.markdown("### 💎 10 Tips to Retain Customers")
        st.write(pd.DataFrame({"Customer Retention Tips": retention_tips}))

# --- FOOTER ---
st.markdown("---")
st.caption("Made by Anmol | Powered by Logistic Regression + Streamlit")
