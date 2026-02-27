import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(data_path='loan.csv'):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please download the dataset.")
        return

    print("Loading dataset...")
    # Loading a subset if it's too large, or full if possible
    try:
        df = pd.read_csv(data_path, low_memory=False, nrows=100000)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("Preprocessing data...")
    # Target Variable Mapping
    bad_statuses = ['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)']
    df['target'] = df['loan_status'].apply(lambda x: 1 if x in bad_statuses else 0)

    # Feature Selection (subset of common features)
    features = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 
        'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 
        'purpose', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 
        'pub_rec', 'revol_bal', 'revol_util', 'total_acc'
    ]
    
    # Filter to only existing columns
    features = [f for f in features if f in df.columns]
    X = df[features].copy()
    y = df['target']

    # Handle Missing Values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna(X[col].mode()[0])
        else:
            X[col] = X[col].fillna(X[col].median())

    # Encoding Categorical Variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Model (Optimized for size)...")
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluation
    y_pred = rf_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # Save Model and Preprocessing objects
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open('models/encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
        
    with open('models/features.pkl', 'wb') as f:
        pickle.dump(features, f)

    print("Model and encoders saved to models/ directory.")

if __name__ == "__main__":
    # Check if loan.csv is in the current directory or parent
    if os.path.exists('e:/3rd_year/ai_ann/class mini projects/loan.csv'):
        train_model('e:/3rd_year/ai_ann/class mini projects/loan.csv')
    else:
        train_model('loan.csv')
