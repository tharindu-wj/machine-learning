import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def main():
    # 1. Load Data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # 2. Preprocessing
    # Split dataset into training and testing sets
    # Stratify ensures the class distribution is preserved
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100, stratify=y
    )

    # Scale features
    # Fit on training set only to avoid data leakage
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Visualization (Interactive Cell)
    # %%
    print("Generating visualization...")
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=iris.feature_names)
    X_train_df = pd.concat([X_train_scaled_df, pd.DataFrame(y_train, columns=['class'])], axis=1)
    sns.pairplot(X_train_df, hue='class', height=2)
    plt.show()

    # 4. Model Training
    print("Training KNeighborsClassifier...")
    # n_neighbors=3 as a standard choice
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)

    # 5. Evaluation
    predictions = knn.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = knn.score(X_test_scaled, y_test)
    
    # Alternative error calculation
    errors = (predictions != y_test)
    error_rate = sum(errors) / errors.size

    print("-" * 30)
    print("Model: KNeighborsClassifier (Iris)")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Error Rate: {error_rate:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()

# %%
