# 💳 Credit Card Risk Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://creadet-card-risk-pridiction.onrender.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance machine learning application that predicts loan default risk. This tool helps financial institutions analyze applicant data and determine creditworthiness using an optimized Random Forest model.

---

## 📺 Demo Snapshot
The application features a sidebar for user inputs and a real-time prediction engine:
*   **Safe**: ✅ Applicant is likely to repay the loan.
*   **Risky**: 🚨 High probability of default.

---

## 🚀 Deployment on Render (Step-by-Step)

To get this running on Render without the "Not Found" error, follow these exact settings:

1.  **Create New Web Service**: Connect your GitHub repo.
2.  **Runtime**: `Python`
3.  **Build Command**: `pip install -r requirements.txt`
4.  **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5.  **Environment Variables**:
    *   Add Key: `PYTHON_VERSION`, Value: `3.10.12`

---

## 📊 Project Workflow
1.  **Data Cleaning**: Handled missing values and preprocessed Lending Club data.
2.  **Feature Engineering**: Selected the 18 most impactful features (DTI, Income, Loan Amount, etc.).
3.  **Model Optimization**: Trained a size-optimized Random Forest model to fit under the 100MB GitHub limit.
4.  **Deployment**: Integrated with Streamlit for a responsive web interface.

---

## 📂 Project Structure
```text
├── models/               
│   ├── rf_model.pkl      # Optimized Random Forest Classifier
│   ├── encoders.pkl      # Categorical Label Encoders
│   └── features.pkl      # List of features for prediction
├── app.py                # Streamlit Web Application
├── model_trainer.py      # Data Preprocessing & Training Script
├── requirements.txt      # Stable library dependencies
├── .gitignore            # Excludes massive datasets while allowing models
└── README.md             # Project Documentation
```

## 🛠️ Local Installation
```bash
git clone https://github.com/Abhishek-Maheshwari-778/creadet_card_risk_pridiction.git
cd creadet_card_risk_pridiction
python -m venv .venv
# Activate venv and run:
pip install -r requirements.txt
streamlit run app.py
```

---
**Developed by [Abhishek Maheshwari](https://github.com/Abhishek-Maheshwari-778)**
