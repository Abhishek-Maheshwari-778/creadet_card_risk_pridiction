import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(page_title="Credit Risk Predictor", layout="wide", page_icon="💳")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("💳 Credit Card Risk Prediction")
st.markdown("---")

# Load model and preprocessing objects
@st.cache_resource
def load_resources():
    # Use absolute paths for Render stability
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'rf_model.pkl')
    encoders_path = os.path.join(base_dir, 'models', 'encoders.pkl')
    features_path = os.path.join(base_dir, 'models', 'features.pkl')
    
    if os.path.exists(model_path) and os.path.exists(encoders_path) and os.path.exists(features_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(encoders_path, 'rb') as f:
                encoders = pickle.load(f)
            with open(features_path, 'rb') as f:
                features = pickle.load(f)
            return model, encoders, features
        except Exception as e:
            st.error(f"Error loading model files: {e}")
    return None, None, None

model, encoders, features = load_resources()

# Sidebar Status
st.sidebar.title("App Status")
if model:
    st.sidebar.success("✅ Prediction Engine Live")
else:
    st.sidebar.error("❌ Model Files Missing")
    st.sidebar.info("Please ensure 'models' folder is pushed to GitHub.")

# Main UI Logic
if model is None:
    st.warning("⚠️ The prediction engine is currently in **Demo Mode** because the trained model files weren't found.")
    features = ['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'emp_length', 'home_ownership', 'annual_inc', 'purpose']

# Sidebar for inputs
st.sidebar.header("Enter Applicant Details")

def user_input_features():
    col1, col2 = st.columns(2)
    inputs = {}
    
    with col1:
        inputs['loan_amnt'] = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, value=10000)
        inputs['term'] = st.selectbox("Term", [" 36 months", " 60 months"])
        inputs['int_rate'] = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
        inputs['installment'] = st.number_input("Monthly Installment ($)", min_value=10.0, max_value=2000.0, value=300.0)
        inputs['grade'] = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        
    with col2:
        inputs['annual_inc'] = st.number_input("Annual Income ($)", min_value=1000, max_value=1000000, value=50000)
        inputs['emp_length'] = st.selectbox("Employment Length", ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"])
        inputs['home_ownership'] = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        inputs['purpose'] = st.selectbox("Purpose", ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase", "medical", "small_business", "car", "vacation", "moving", "house", "wedding", "renewable_energy", "educational"])

    # Required background features
    all_possible_features = ['dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'verification_status']
    for f in all_possible_features:
        if f == 'verification_status': inputs[f] = "Not Verified"
        elif f == 'dti': inputs[f] = 15.0
        else: inputs[f] = 0

    return pd.DataFrame([inputs])

input_df = user_input_features()

st.subheader("📋 Applicant Summary")
st.write(input_df)

if st.button("Predict Credit Risk"):
    if model:
        proc_df = input_df.copy()
        for col, le in encoders.items():
            if col in proc_df.columns:
                try:
                    proc_df[col] = le.transform(proc_df[col].astype(str))
                except:
                    proc_df[col] = le.transform([le.classes_[0]])[0]
        
        proc_df = proc_df[features]
        prediction = model.predict(proc_df)
        prediction_proba = model.predict_proba(proc_df)
        
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        if prediction[0] == 0:
            st.success("✅ **SAFE**: This applicant is likely to repay the loan.")
        else:
            st.error("🚨 **RISKY**: High probability of default detected.")
            
        st.info(f"**Model Confidence:** {max(prediction_proba[0])*100:.2f}%")
    else:
        st.error("Prediction engine is offline. Demo result based on interest rate logic:")
        res = "RISKY" if input_df['int_rate'].values[0] > 18 else "SAFE"
        st.write(f"RESULT: {res}")

st.sidebar.markdown("---")
st.sidebar.info("AI Mini Project Module-1")
