import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Credit Guard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM DARK MODE DESIGN SYSTEM ---
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
    /* Global Styles */
    * { font-family: 'Outfit', sans-serif; }
    .stApp { background: radial-gradient(circle at top right, #1e293b, #0f172a); color: #f1f5f9; }
    /* Glassmorphism Sidebar */
    section[data-testid="stSidebar"] { background-color: rgba(15, 23, 42, 0.8) !important; backdrop-filter: blur(10px); border-right: 1px solid rgba(255, 255, 255, 0.1); }
    /* Headers */
    h1 { font-weight: 700 !important; letter-spacing: -0.02em !important; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem !important; }
    h2, h3, p, label { color: #e2e8f0 !important; }
    /* Cards/Containers */
    .stMarkdown div[data-testid="stMarkdownContainer"] p { font-weight: 400; opacity: 0.9; }
    /* Input Fields Styling */
    .stNumberInput, .stSelectbox, .stTextInput, .stSlider { background-color: rgba(30, 41, 59, 0.5); border-radius: 12px; padding: 5px; border: 1px solid rgba(255, 255, 255, 0.05); }
    .stNumberInput input, .stSelectbox div { color: white !important; background-color: transparent !important; }
    /* Buttons */
    .stButton>button { width: 100%; background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important; color: white !important; border: none !important; padding: 0.75rem 1.5rem !important; border-radius: 12px !important; font-weight: 600 !important; font-size: 1.1rem !important; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06) !important; text-transform: uppercase; letter-spacing: 0.05em; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.2) !important; background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%) !important; }
    /* Result Containers */
    .stAlert { border-radius: 16px !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; backdrop-filter: blur(10px); }
    /* Custom Dataframe */
    [data-testid="stDataFrame"] { background-color: rgba(15, 23, 42, 0.4); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1); }
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    /* Decoration */
    .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; background: rgba(37, 99, 235, 0.2); color: #38bdf8; border: 1px solid rgba(37, 99, 235, 0.3); margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIC & MODEL LOADING ---
@st.cache_resource
def load_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'rf_model.pkl')
    encoders_path = os.path.join(base_dir, 'models', 'encoders.pkl')
    features_path = os.path.join(base_dir, 'models', 'features.pkl')
    
    if os.path.exists(model_path) and os.path.exists(encoders_path) and os.path.exists(features_path):
        try:
            with open(model_path, 'rb') as f: model = pickle.load(f)
            with open(encoders_path, 'rb') as f: encoders = pickle.load(f)
            with open(features_path, 'rb') as f: features = pickle.load(f)
            return model, encoders, features
        except Exception: pass
    return None, None, None

model, encoders, features = load_resources()

# --- SIDEBAR UI ---
with st.sidebar:
    st.markdown('<div class="badge">SYSTEM READY</div>', unsafe_allow_html=True)
    st.title("Credit Guard AI")
    st.info("AI-powered financial risk assessment engine.")
    
    st.markdown("---")
    st.subheader("�‍💻 Development")
    st.markdown("""
        <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);'>
            <p style='margin-bottom: 10px; font-size: 0.9rem;'><b>Created by:</b><br>
            <a href='https://github.com/Abhishek-Maheshwari-778' target='_blank' style='color: #38bdf8; text-decoration: none; font-weight: 600;'>Abhishek Maheshwari 🔗</a></p>
            <p style='margin-bottom: 10px; font-size: 0.9rem;'><b>Mentor:</b><br>
            <a href='https://github.com/Amarjeet9305' target='_blank' style='color: #38bdf8; text-decoration: none; font-weight: 600;'>Amarjeet 🔗</a></p>
            <p style='margin-top: 10px; font-size: 0.8rem; color: #818cf8; font-style: italic;'>✨ Created using Vibe Coding</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("AI Mini Project • 2026")

# --- MAIN UI ---
col_main = st.container()

with col_main:
    st.title("Credit Risk Analysis")
    st.markdown("Assess the probability of loan default using multi-variate risk scoring.")
    
    st.markdown("### 📝 Application Form")
    
    # Organize inputs into tabs or clean grid
    with st.expander("👤 Applicant Information", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            annual_inc = st.number_input("Annual Income ($)", min_value=1000, value=55000, step=1000, help="Gross yearly income")
            emp_length = st.selectbox("Employment Length", ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"])
        with c2:
            home_ownership = st.selectbox("Home Ownership Status", ["MORTGAGE", "RENT", "OWN", "OTHER"])
            verification_status = st.selectbox("Verification Status", ["Verified", "Source Verified", "Not Verified"])

    with st.expander("💰 Loan Details", expanded=True):
        c3, c4 = st.columns(2)
        with c3:
            loan_amnt = st.number_input("Loan Amount Requested ($)", min_value=500, value=12000, step=500)
            term = st.selectbox("Loan Term", [" 36 months", " 60 months"])
            int_rate = st.slider("Interest Rate (%)", 5.0, 35.0, 11.5, step=0.1)
        with c4:
            grade = st.selectbox("Lending Grade Score", ["A", "B", "C", "D", "E", "F", "G"])
            purpose = st.selectbox("Purpose of Loan", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "medical", "small_business", "car", "other"])
            installment = st.number_input("Monthly Installment ($)", min_value=10.0, value=350.0)

    # Prediction Logic
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Calculate Risk Score"):
        inputs = {
            'loan_amnt': loan_amnt, 'term': term, 'int_rate': int_rate, 
            'installment': installment, 'grade': grade, 'emp_length': emp_length,
            'home_ownership': home_ownership, 'annual_inc': annual_inc, 
            'purpose': purpose, 'verification_status': verification_status
        }
        
        # Add background defaults
        bg_features = {'dti': 15.0, 'delinq_2yrs': 0, 'inq_last_6mths': 0, 'open_acc': 10, 'pub_rec': 0, 'revol_bal': 5000, 'revol_util': 30, 'total_acc': 15}
        inputs.update(bg_features)
        
        input_df = pd.DataFrame([inputs])
        
        with st.spinner("Analyzing financial vectors..."):
            if model:
                proc_df = input_df.copy()
                for col, le in encoders.items():
                    if col in proc_df.columns:
                        try: proc_df[col] = le.transform(proc_df[col].astype(str))
                        except: proc_df[col] = le.transform([le.classes_[0]])[0]
                
                proc_df = proc_df[features if features else proc_df.columns]
                prediction = model.predict(proc_df)
                prediction_proba = model.predict_proba(proc_df)
                
                st.markdown("### 🎯 Risk Assessment Result")
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    if prediction[0] == 0:
                        st.success(f"### Score: SAFE \n No immediate risk patterns detected for ${loan_amnt:,.0f} loan.")
                    else:
                        st.error(f"### Score: HIGH RISK \n Alert: Financial instability indicators detected.")
                
                with res_col2:
                    confidence = max(prediction_proba[0]) * 100
                    st.metric("Model Confidence", f"{confidence:.1f}%")
                    st.progress(confidence / 100)
            else:
                st.warning("Prediction engine offline. Running fallback heuristics...")
                is_risky = int_rate > 17 or grade in ["E", "F", "G"]
                if not is_risky:
                    st.success(f"### Score: SAFE (Heuristic Prediction)")
                else:
                    st.error(f"### Score: RISKY (Heuristic Prediction)")

# Styling bottom elements
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 10px; color: #64748b;">
        <p style="font-size: 1rem; margin-bottom: 0;">Built with ❤️ & ⚡ <b>Vibe Coding</b></p>
        <p style="font-size: 0.8rem; opacity: 0.7;">Secure SSL Encrypted • Powered by Random Forest Engine</p>
    </div>
""", unsafe_allow_html=True)
