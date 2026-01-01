# Machine Learning & Neural Networks Repository

This repository contains a comprehensive collection of Machine Learning and Neural Networks examples, structured for clarity and ease of learning. The code is organized by algorithm type and task, with all implementations derived from standard Scikit-Learn practices.

## Repository Structure

The code is organized into a hierarchical directory structure:

```text
ml_repo/
├── data/                                   # Datasets (e.g., California Housing, Genomic Data)
├── supervised/
│   ├── classification/
│   │   ├── knn/                            # K-Nearest Neighbors Classification
│   │   ├── logistic_regression/            # Logistic Regression (w/ L1/L2 regularization)
│   │   ├── decision_trees/                 # Decision Trees (w/ visualization)
│   │   ├── random_forest/                  # Random Forest Classification (w/ CV)
│   │   └── ensemble/                       # Ensemble Methods (Bagging, AdaBoost)
│   └── regression/
│       ├── knn/                            # KNN Regression
│       ├── linear_models/                  # Linear, Lasso, Ridge, ElasticNet
│       └── random_forest/                  # Random Forest Regression (w/ GridSearch)
├── unsupervised/
│   ├── clustering/
│   │   └── kmeans/                         # K-Means Clustering
│   └── dimensionality_reduction/
│       └── pca/                            # Principal Component Analysis (PCA)
├── neural_networks/                        # (Future Scope) Deep Learning models
│   ├── basic/
│   ├── cnn/
│   └── rnn/
├── requirements.txt                        # Project dependencies
└── README.md                               # This documentation
```

## Implemented Algorithms

### Supervised Learning
*   **Classification**
    *   **KNN**: `supervised/classification/knn/iris_classification.py`
    *   **Logistic Regression**: `supervised/classification/logistic_regression/genome_prediction.py`
    *   **Decision Trees**: `supervised/classification/decision_trees/iris_classification.py`
    *   **Random Forest**: `supervised/classification/random_forest/breast_cancer_classification.py`
    *   **Bagging**: `supervised/classification/ensemble/bagging_iris.py`
    *   **AdaBoost**: `supervised/classification/ensemble/adaboost_iris.py`
*   **Regression**
    *   **KNN**: `supervised/regression/knn/california_housing.py`
    *   **Linear/Lasso/Ridge/ElasticNet**: `supervised/regression/linear_models/student_performance.py`
    *   **Random Forest**: `supervised/regression/random_forest/california_housing_regression.py`

### Unsupervised Learning
*   **Dimensionality Reduction**
    *   **PCA**: `unsupervised/dimensionality_reduction/pca/iris_pca.py`
*   **Clustering**
    *   **K-Means**: `unsupervised/clustering/kmeans/iris_kmeans.py`

## Getting Started

### Prerequisites
*   Python 3.8+
*   pip

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd ml_repo
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Each algorithm is implemented as a standalone Python script. Results are printed to the console, and visualizations are available via interactive mode.

### Running a Script
Navigate to the root directory and run any script using `python`:

```bash
# Example: Run Decision Tree Classification
python supervised/classification/decision_trees/iris_classification.py
```

### Visualizations
The scripts include interactive visualization cells marked with `# %%`.
*   **VS Code / PyCharm**: You can run these cells directly in your editor to see plots (e.g., decision trees, scatter plots) inline, similar to a Jupyter Notebook.
*   **Standard Execution**: `plt.show()` is called at the end of visualization blocks, so running the script normally will also pop up a window with the plot.

## Code Style
*   **Structure**: All scripts follow a consistent structure: Load Data -> Preprocessing -> Model Training -> Evaluation -> Visualization.
*   **Standardization**: Output metrics (Accuracy, R^2, etc.) are printed in a standardized format.
