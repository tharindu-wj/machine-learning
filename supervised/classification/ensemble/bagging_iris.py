import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
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
    # Placeholder for visualization (can add decision boundary plot if needed)
    # print("Visualization skipped for Bagging example.")

    # 4. Model Training
    print("Training BaggingClassifier (Base: DecisionTree)...")
    
    # Create Bagging Classifier
    # Base estimator is Decision Tree by default
    bagging_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        random_state=42
    )
    bagging_model.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = bagging_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Comparison with single Decision Tree
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)
    tree_accuracy = tree_model.score(X_test, y_test)

    print("-" * 30)
    print("Model: BaggingClassifier (Iris)")
    print("-" * 30)
    print(f"Bagging Classifier Accuracy: {accuracy:.4f}")
    print(f"Single Decision Tree Accuracy: {tree_accuracy:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
