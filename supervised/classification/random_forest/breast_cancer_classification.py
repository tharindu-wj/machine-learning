import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # 1. Load Data
    dataset = datasets.load_breast_cancer()
    X = dataset.data
    y = dataset.target
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # 2. Preprocessing
    # No specific preprocessing beyond what's needed for the model (Random Forest is robust to scaling)
    
    # 3. Visualization
    # %%
    # Skipping complex high-dimensional visualization for this template
    # but providing placeholder
    # print("Visualization skipped for Breast Cancer dataset.")

    # 4. Model Training & Evaluation (Cross-Validation)
    print("Starting 10-Fold Cross Validation with RandomForestClassifier...")
    
    # Define RandomForestClassifier
    rf_model = RandomForestClassifier(
        n_estimators=50, 
        min_samples_leaf=5, 
        criterion='gini', 
        random_state=42, 
        max_features='sqrt'
    )

    # Define StratifiedKFold
    skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    K = 10
    cv_accuracy = np.zeros(K)
    cv_precision = np.zeros(K)
    cv_recall = np.zeros(K)
    cv_f1 = np.zeros(K)

    for counter, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
        X_train_kfold = X[train_idx, :]
        y_train_kfold = y[train_idx]
        X_test_kfold = X[test_idx, :]
        y_test_kfold = y[test_idx]

        rf_model.fit(X_train_kfold, y_train_kfold)
        y_pred = rf_model.predict(X_test_kfold)

        # Calculate metrics
        cv_accuracy[counter] = accuracy_score(y_test_kfold, y_pred)
        cv_precision[counter] = precision_score(y_test_kfold, y_pred)
        cv_recall[counter] = recall_score(y_test_kfold, y_pred)
        cv_f1[counter] = f1_score(y_test_kfold, y_pred)

    # 5. Evaluation Output
    print("-" * 30)
    print("Model: RandomForestClassifier (Breast Cancer)")
    print("Method: 10-Fold Cross Validation")
    print("-" * 30)
    print(f"Average Accuracy:  {cv_accuracy.mean():.4f}")
    print(f"Average Precision: {cv_precision.mean():.4f}")
    print(f"Average Recall:    {cv_recall.mean():.4f}")
    print(f"Average F1 Score:  {cv_f1.mean():.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
