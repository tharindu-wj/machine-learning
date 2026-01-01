import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

def main():
    # 1. Load Data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target # True labels for comparison
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # 2. Preprocessing
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Model Training (KMeans)
    print("Training KMeans Clustering...")
    # We know there are 3 classes in Iris
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    y_pred = kmeans.labels_

    # 4. Visualization (Interactive Cell)
    # %%
    print("Generating KMeans visualization...")
    plt.figure(figsize=(8, 6))
    
    # Plot data points colored by cluster
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.6, label='Data Points')
    
    # Plot centroids
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=200, c='red', marker='X', label='Centroids')
    
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    plt.title('KMeans Clustering on Iris (First 2 Features)')
    plt.legend()
    plt.show()

    # 5. Evaluation
    sil_score = silhouette_score(X_scaled, y_pred)
    ari_score = adjusted_rand_score(y, y_pred)

    print("-" * 30)
    print("Algorithm: KMeans Clustering (Iris)")
    print("Parameters: n_clusters=3")
    print("-" * 30)
    print(f"Silhouette Score:    {sil_score:.4f}")
    print(f"Adjusted Rand Score: {ari_score:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
