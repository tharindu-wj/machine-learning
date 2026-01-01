import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def main():
    # 1. Load Data
    housing = datasets.fetch_california_housing()
    X = housing.data
    y = housing.target
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # 2. Preprocessing
    # Using Grid Search approach from Module 5 Checkpoint 3
    print("Setting up Grid Search...")
    np.random.seed(42)

    # Split Data (Train/Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Visualization
    # %%
    # print("Visualization skipped for Grid Search example.")

    # 4. Model Training & Hyperparameter Tuning
    
    # Define Regressor
    rf = RandomForestRegressor(random_state=42)
    
    # Define Hyperparameters
    param_grid = {
        'n_estimators': [10, 50, 100], 
        'min_samples_leaf': [2, 10]
    }
    
    # Define KFold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define GridSearchCV
    print("Starting GridSearchCV (This may take a moment)...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=kfold,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # 5. Evaluation
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test_scaled, y_test)
    
    print("-" * 30)
    print("Model: RandomForestRegressor (California Housing)")
    print("Method: GridSearchCV")
    print("-" * 30)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Best CV R^2 Score:    {grid_search.best_score_:.4f}")
    print(f"Test Set R^2 Score:   {test_score:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
