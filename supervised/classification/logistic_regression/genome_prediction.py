import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def main():
    # 1. Load Data
    # Define path to dataset with fallback
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/GenomicData.csv')
    
    if not os.path.exists(data_path):
        data_path = 'data/GenomicData.csv' # Fallback for relative run
        
    print(f"Loading data from: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: Dataset not found. Please ensure 'GenomicData.csv' is in the 'data' directory.")
        return

    # 2. Preprocessing
    # Drop redundant columns (indices 0 and 1)
    df_reduced = df.drop(df.columns[[0, 1]], axis=1)
    print(f"Dataset shape after dropping initial columns: {df_reduced.shape}")

    # Set seed
    np.random.seed(42)

    # Handle missing values: Drop columns with '?' (which becomes NaN if replaced, or if dropped)
    # Replicating logic: Drop columns with NaNs seems to be the preferred method from previous work
    # First replace '?' with NaN to ensure dropna works if they aren't already NaN
    df_reduced = df_reduced.replace('?', np.nan)
    df_reduced_no_miss = df_reduced.dropna(axis=1)
    print(f"Dataset shape after dropping columns with missing values: {df_reduced_no_miss.shape}")

    y = df_reduced_no_miss["ClassType"]
    X = df_reduced_no_miss.drop(columns=["ClassType"])

    # Split dataset
    # Stratify ensures class balance in train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Visualization (Interactive Cell)
    # %%
    # No visualization code provided for this task, but placeholder can exist.
    # print("Visualization skipped for high-dimensional genomic data.")

    # 4. Model Training & 5. Evaluation
    print("-" * 30)
    print("Model: Logistic Regression (Genomic Data)")
    print("-" * 30)

    # Train with C=0.05
    print("Training with C=0.05 (L1 penalty)...")
    logr_005 = LogisticRegression(penalty="l1", solver="liblinear", C=0.05)
    logr_005.fit(X_train_scaled, y_train)
    score_005 = logr_005.score(X_test_scaled, y_test)
    num_features_005 = np.sum(logr_005.coef_ != 0)
    print(f"Accuracy (C=0.05): {score_005:.4f}")
    print(f"Selected Features: {num_features_005}")

    # Train with C=10
    print("\nTraining with C=10 (L1 penalty)...")
    logr_10 = LogisticRegression(penalty="l1", solver="liblinear", C=10)
    logr_10.fit(X_train_scaled, y_train)
    score_10 = logr_10.score(X_test_scaled, y_test)
    num_features_10 = np.sum(logr_10.coef_ != 0)
    print(f"Accuracy (C=10):   {score_10:.4f}")
    print(f"Selected Features: {num_features_10}")
    print("-" * 30)

if __name__ == "__main__":
    main()
