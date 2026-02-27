import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

st.title("💳 Credit Risk Prediction")
st.markdown("""
Predict the likelihood of a loan applicant defaulting based on their financial history and loan details.
""")

# Load model and preprocessing objects
def load_resources():
    # Try both relative and absolute paths for Render stability
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'rf_model.pkl')
    encoders_path = os.path.join(base_dir, 'models', 'encoders.pkl')
    features_path = os.path.join(base_dir, 'models', 'features.pkl')
    
    if os.path.exists(model_path) and os.path.exists(encoders_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(encoders_path, 'rb') as f:
                encoders = pickle.load(f)
            with open(features_path, 'rb') as f:
                features = pickle.load(f)
            return model, encoders, features
        except Exception as e:
            st.error(f"Error loading pickle files: {e}")
            return None, None, None
    
    # Debug info for the user if files are missing
    if not os.path.exists(os.path.join(base_dir, 'models')):
        st.error(f"Directory 'models' not found in {base_dir}")
    else:
        st.write(f"Models directory found, but files might be missing. Files present: {os.listdir(os.path.join(base_dir, 'models'))}")
        
    return None, None, None

model, encoders, features = load_resources()

if model is None:
    st.warning("⚠️ Model files not found! Please run `model_trainer.py` first with the dataset.")
    # Show a mockup UI anyway
    st.info("Showing mockup UI for demonstration.")
    features = ['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'emp_length', 'home_ownership', 'annual_inc', 'purpose']

# Sidebar for inputs
st.sidebar.header("User Input Features")

def user_input_features():
    inputs = {}
    
    col1, col2 = st.columns(2)
    
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

    # Add remaining features with default values if not explicitly in UI
    all_possible_features = [
        'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'verification_status'
    ]
    for f in all_possible_features:
        if f not in inputs:
            if f == 'verification_status':
                inputs[f] = "Not Verified"
            else:
                inputs[f] = 0

    return pd.DataFrame([inputs])

input_df = user_input_features()

st.subheader("Applicant Summary")
st.write(input_df)

if st.button("Predict Risk"):
    if model:
        # Preprocess input
        proc_df = input_df.copy()
        for col, le in encoders.items():
            if col in proc_df.columns:
                try:
                    proc_df[col] = le.transform(proc_df[col].astype(str))
                except:
                    # Handle unseen labels by choosing a default or most frequent
                    proc_df[col] = le.transform([le.classes_[0]])[0]
        
        # Ensure feature order matches trainer
        proc_df = proc_df[features]
        
        prediction = model.predict(proc_df)
        prediction_proba = model.predict_proba(proc_df)
        
        st.subheader("Prediction Result")
        if prediction[0] == 0:
            st.success("✅ **SAFE**: Loan is likely to be repaid.")
        else:
            st.error("🚨 **RISKY**: High probability of default.")
            
        st.write(f"Confidence: {max(prediction_proba[0])*100:.2f}%")
    else:
        st.error("Model not trained yet. Please provide 'loan.csv' and run the trainer script.")
        # Mock result for logic test
        res = "RISKY" if input_df['int_rate'].values[0] > 15 else "SAFE"
        st.write(f"MOCK RESULT: {res}")

st.sidebar.markdown("---")
st.sidebar.info("This app is part of the AI Mini Project Module-1.")
