import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

def main():
    # 1. Load Data
    # Define path to dataset
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/student-mat.csv')
    
    if not os.path.exists(data_path):
        data_path = 'data/student-mat.csv' # Fallback
    
    print(f"Loading data from: {data_path}")
    
    try:
        df = pd.read_csv(data_path, sep=";")
    except FileNotFoundError:
        print("Error: Dataset not found. Please ensure 'student-mat.csv' is in the 'data' directory.")
        return

    # 2. Preprocessing
    # Split into labels and input features
    # Target is G3 (final grade)
    if not 'G3' in df.columns:
        print("Error: Target column 'G3' not found in dataset")
        return

    y = df["G3"]
    X = df.drop(columns=["G3"])

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Encode categorical features using One-Hot Encoding
    X_train_encoded = pd.get_dummies(X_train)
    X_test_encoded = pd.get_dummies(X_test)

    # Align columns to ensure both sets have the same features (handle missing categories in test set)
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

    # Scale features
    scaler = StandardScaler()
    scaler.fit(X_train_encoded)
    X_train_scaled = scaler.transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # 3. Visualization
    # %%
    # print("Visualization skipped for Student Performance.")

    # 4. Model Training & Evaluation
    print("-" * 30)
    print("Model: Linear Models (Student Performance)")
    print("-" * 30)

    # Train Linear Regression model
    model_lin = LinearRegression()
    model_lin.fit(X_train_scaled, y_train)
    score_lin = model_lin.score(X_test_scaled, y_test)
    print(f"LinearRegression R^2: {score_lin:.4f}")

    # Train Lasso model (L1 regularization)
    model_lass = Lasso(alpha=0.1)
    model_lass.fit(X_train_scaled, y_train)
    score_lass = model_lass.score(X_test_scaled, y_test)
    print(f"Lasso R^2:            {score_lass:.4f}")

    # Train Ridge model (L2 regularization)
    model_ridg = Ridge(alpha=0.1)
    model_ridg.fit(X_train_scaled, y_train)
    score_ridg = model_ridg.score(X_test_scaled, y_test)
    print(f"Ridge R^2:            {score_ridg:.4f}")

    # Train ElasticNet model (L1 + L2 regularization)
    model_elastic = ElasticNet(alpha=0.1)
    model_elastic.fit(X_train_scaled, y_train)
    score_elastic = model_elastic.score(X_test_scaled, y_test)
    print(f"ElasticNet R^2:       {score_elastic:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
