import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

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

    # 3. Visualization (Interactive Cell)
    # %%
    print("Generating decision tree visualization...")
    plt.figure(figsize=(12, 8))
    # Using a max_depth=3 tree for visualization clarity
    viz_model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
    viz_model.fit(X_train, y_train)
    plot_tree(viz_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
    plt.title("Decision Tree Visualization (Max Depth=3)")
    plt.show()

    # 4. Model Training
    print("Training DecisionTreeClassifier...")
    # Training a simple model as per original intent (max_depth=1 was used for illustration in notebook)
    # But let's make it more generally useful with a slightly deeper tree or default
    model_1 = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
    model_1.fit(X_train, y_train)

    # 5. Evaluation
    accuracy = model_1.score(X_test, y_test)

    print("-" * 30)
    print("Model: DecisionTreeClassifier (Iris)")
    print("Hyperparameters: criterion='entropy', max_depth=3")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
