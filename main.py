import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
import requests
from io import StringIO

# Scikit-learn modules for Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# 1. Data Loading (UCI Phishing Websites Dataset)
# ---------------------------------------------------------
def load_data():
    """
    Loads the UCI Phishing Websites dataset.
    This function handles the .arff format common to UCI datasets.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
    print(f"Downloading dataset from: {url}...")
    
    response = requests.get(url)
    data, meta = arff.loadarff(StringIO(response.text))
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame(data)
    
    # Decode byte strings to normal strings (utf-8) if necessary
    # The UCI dataset often loads as byte strings (b'1', b'-1')
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8').astype(int)
            
    print(f"Dataset Loaded Successfully. Shape: {df.shape}")
    return df

# ---------------------------------------------------------
# 2. Data Preprocessing
# ---------------------------------------------------------
def preprocess_data(df):
    """
    Splits features/target and normalizes the data.
    """
    # The target variable is usually the last column 'Result'
    # 1 = Legitimate, -1 = Phishing
    X = df.iloc[:, :-1] # All columns except the last
    y = df.iloc[:, -1]  # The last column
    
    # Split into Train (70%) and Test (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Feature Scaling (Crucial for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# ---------------------------------------------------------
# 3. Model Training & Evaluation
# ---------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name):
    """
    Predicts and calculates metrics for a given model.
    """
    print(f"--- Evaluating {model_name} ---")
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return y_pred

# ---------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # A. Load Data
    df = load_data()
    
    # B. Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # C. Model 1: Random Forest
    print("\n1. Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Feature Importance (Optional but recommended for the paper)
    importances = rf_model.feature_importances_
    # (Code to plot importances could go here)
    
    # D. Model 2: Support Vector Machine (SVM)
    print("\n2. Training Support Vector Machine (SVM)...")
    # Kernel='rbf' allows for non-linear separation
    svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test, "SVM")
    
    print("\nExecution Complete.")
