import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

# Custom CSS for Professional Look and Fixed Visibility
st.markdown("""
    <style>
    /* Force main title and headers to be visible */
    h1, h2, h3, p, span, label {
        color: #1E293B !important; /* Dark Slate color */
    }
    
    /* Main Background */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p, 
    section[data-testid="stSidebar"] label {
        color: #F8FAFC !important;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #2563EB !important;
        color: white !important;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    
    /* Input field styling for visibility */
    .stNumberInput input, .stSelectbox div {
        color: #1E293B !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Credit Card Risk Prediction")
st.markdown("Assess applicant creditworthiness based on financial history and loan details.")
st.markdown("---")

# Load model and preprocessing objects
@st.cache_resource
def load_resources():
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
        except Exception:
            pass
    return None, None, None

model, encoders, features = load_resources()

# Sidebar Status
st.sidebar.title("System Status")
if model:
    st.sidebar.info("Prediction Engine: ACTIVE")
else:
    st.sidebar.warning("Prediction Engine: OFFLINE (Demo Mode)")

# Sidebar for inputs
st.sidebar.header("Applicant Details")

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

st.subheader("Data Summary")
st.dataframe(input_df, use_container_width=True)

if st.button("RUN PREDICTION"):
    if model:
        proc_df = input_df.copy()
        for col, le in encoders.items():
            if col in proc_df.columns:
                try:
                    proc_df[col] = le.transform(proc_df[col].astype(str))
                except:
                    proc_df[col] = le.transform([le.classes_[0]])[0]
        
        proc_df = proc_df[features if features else proc_df.columns]
        prediction = model.predict(proc_df)
        prediction_proba = model.predict_proba(proc_df)
        
        st.markdown("---")
        st.subheader("Results")
        if prediction[0] == 0:
            st.success("SAFE: High probability of repayment.")
        else:
            st.error("RISKY: Potential for default detected.")
            
        st.info(f"Analysis Confidence: {max(prediction_proba[0])*100:.2f}%")
    else:
        st.warning("Prediction engine offline. Demo logic active:")
        res = "RISKY" if input_df['int_rate'].values[0] > 18 else "SAFE"
        st.write(f"PROJECTION: {res}")

st.sidebar.markdown("---")
st.sidebar.caption("AI Mini Project Module-1")
