import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    # 1. Load Data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # 2. Preprocessing
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Model Training (PCA)
    print("Applying PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 4. Visualization (Interactive Cell)
    # %%
    print("Generating PCA visualization...")
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA on Iris Dataset')
    plt.colorbar(scatter, label='Target')
    plt.show()

    # 5. Evaluation
    print("-" * 30)
    print("Algorithm: PCA (Iris)")
    print("-" * 30)
    print(f"Original Shape: {X.shape}")
    print(f"Reduced Shape:  {X_pca.shape}")
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
