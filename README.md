# 💳 Credit Card Risk Prediction

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://creadet-card-risk-pridiction.onrender.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning-powered web application that predicts the credit risk of loan applications. It classifies applicants into **Safe** or **Risky** categories using advanced classification algorithms.

## 🚀 Live Demo
You can access the live application here: [Credit Risk Predictor on Render](https://creadet-card-risk-pridiction.onrender.com) (Update this link after deployment)

---

## 📌 Project Overview
The goal of this project is to provide a tool for financial institutions to assess the creditworthiness of applicants. By analyzing historical loan data, the model identifies patterns associated with defaults and successful repayments.

### Key Features
- **Interactive Web UI:** Built with Streamlit for a seamless user experience.
- **Robust ML Pipeline:** Includes data cleaning, feature engineering, and model evaluation.
- **Multiple Models:** Explores Random Forest, SVM, and XGBoost to ensure the best performance.
- **Real-time Prediction:** Get instant results by entering applicant details.

---

## 📊 Dataset
The project uses the **Lending Club Loan Data**, a comprehensive dataset containing thousands of loan records.
- **Source:** [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset)
- **Primary file:** `loan.csv`

> [!IMPORTANT]
> Due to GitHub file size limits (100MB), the raw dataset (`loan.csv`) is ignored via `.gitignore`. Please download it manually from Kaggle if you wish to retrain the model.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Abhishek-Maheshwari-778/creadet_card_risk_pridiction.git
cd creadet_card_risk_pridiction
```

### 2. Set Up Environment
Create a virtual environment and install the required packages:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Application Locally
```bash
streamlit run app.py
```

---

## 🌐 Deployment

### Deploying to Render
1. Create a free account on [Render](https://render.com/).
2. Create a new **Web Service**.
3. Connect your GitHub repository: `Abhishek-Maheshwari-778/creadet_card_risk_pridiction`.
4. Configure the settings:
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py`

### Deploying to Streamlit Cloud
1. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **New app**.
3. Select your repository, branch (`main`), and main file path (`app.py`).
4. Click **Deploy!**

---

## 📂 Project Structure
```text
├── models/               # Saved trained models and encoders
├── app.py                # Main Streamlit application
├── model_trainer.py      # Script to train and save models
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
├── LICENSE               # MIT License
└── .gitignore            # Files excluded from GitHub
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**Developed by [Abhishek Maheshwari](https://github.com/Abhishek-Maheshwari-778)**
