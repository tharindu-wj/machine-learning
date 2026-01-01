import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

def main():
    # 1. Load Data
    housing = datasets.fetch_california_housing()
    X = housing.data
    y = housing.target
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # 2. Preprocessing
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Visualization (Interactive Cell)
    # %%
    # No specific visualization requested for regression but a pairplot or correlation matrix could be useful
    # print("Visualization skipped for regression example.")

    # 4. Model Training
    print("Training KNeighborsRegressor...")
    # Using parameters from original notebook logic (3 neighbors, euclidean distance)
    # Weights='distance' gives more weight to closer neighbors
    knn_model = KNeighborsRegressor(n_neighbors=3, weights='distance', metric='euclidean')
    knn_model.fit(X_train_scaled, y_train)

    # 5. Evaluation
    score = knn_model.score(X_test_scaled, y_test)

    print("-" * 30)
    print("Model: KNeighborsRegressor (California Housing)")
    print("Parameters: n_neighbors=3, weights='distance', metric='euclidean'")
    print("-" * 30)
    print(f"R^2 Score: {score:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
