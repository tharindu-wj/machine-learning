import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def main():
    # 1. Load Data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # 2. Preprocessing
    # Split dataset
    # Stratify ensures the class distribution is preserved
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Visualization
    # %%
    # Placeholder for visualization
    # print("Visualization skipped for AdaBoost example.")

    # 4. Model Training
    print("Training AdaBoostClassifier (Base: DecisionTree)...")
    
    # Create AdaBoost Classifier
    # Base estimator is Decision Tree (max_depth=1) by default
    adaboost_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        random_state=42,
        algorithm='SAMME' # 'SAMME.R' is deprecated in newer sklearn versions
    )
    adaboost_model.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = adaboost_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("-" * 30)
    print("Model: AdaBoostClassifier (Iris)")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
