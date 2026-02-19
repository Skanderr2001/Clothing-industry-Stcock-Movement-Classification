# Stock Movement Classification in the Clothing Industry

## Project Overview

This project, developed by Skander Radhouane, focuses on building a classification model to predict the stock movement of raw materials and consumables within the clothing industry. The primary goal is to categorize items as either "Moving" or "Non/Slow Moving" to optimize inventory management and reduce potential losses from dormant stock.

## Dataset

The analysis utilizes three distinct datasets, likely sourced from an internal inventory system, covering various aspects of the clothing manufacturing process:

*   **Supply Data**: Contains information on general supplies such as labels, rivet nails, leather, and sewing threads. Key attributes include `family`, `id`, `supply name`, `stock location`, `quantity`, `latest movement date`, `unit price`, `currency`, `total price in TND`, `Number of Days Since latest Movement`, `Stock Movement Class`, `Dormant Stock Value in TND`, `Current Stock Value in TND`, and `Provision`.
*   **Fabric Data**: Details various types of fabrics used in production. Attributes include `fabric_id`, `Fabric Name`, `Market Quantity`, `In process Quantity`, `Total Quantity`, `total quantity2`, `Unit Price`, `currency`, `total price in TND`, `last_output`, `Latest Movement Date`, `Number of days since last Movement`, `Stock Movement Class`, `Dormant Stock Value`, `Current Stock Value`, and `Provision`.
*   **Spare Part Data**: Encompasses data on spare parts like needles and other machine components. Attributes include `Band`, `Article number`, `Article name`, `quantity`, `unit price in tnd`, `total price in tnd`, `latest movement date`, `Latest Movement date Adjusted by ECOVIS`, `Number of Days Since last Movement`, `Stock Movement Class`, `DORMANT STOCK value`, `Current Stock Value`, and `Provision`.

## Problem Statement

The core problem addressed is the classification of inventory items into two categories: "Moving" and "Non/Slow Moving." This classification is crucial for identifying slow-moving or dormant stock, which can lead to increased holding costs, obsolescence, and reduced liquidity. By accurately predicting stock movement, businesses can make informed decisions regarding purchasing, promotions, and disposal.

## Methodology

The project follows a standard machine learning pipeline, encompassing data loading, preprocessing, exploratory data analysis, model building, and evaluation.

### 1. Data Loading and Initial Exploration

Data from Excel files (`Supply.xlsx`, `Fabric.xlsx`, `Spare Part.xlsx`) is loaded into Pandas DataFrames. Initial steps involve displaying the first few rows of each dataset, checking for missing values, and generating basic summary statistics to understand the data distribution and identify potential issues.

### 2. Data Preprocessing

*   **Missing Value Handling**: Rows with missing values in critical columns, particularly `quantity`, are dropped to ensure data integrity for model training.
*   **Feature Engineering**: 
    *   Column names are standardized across the different datasets to facilitate merging. 
    *   The three datasets (Supply, Fabric, Spare Part) are merged into a single comprehensive DataFrame named `Consumables_and_Raw_Materials_Data`.
    *   A new target variable, `The Movement`, is created based on the `Stock Movement Class` column. Items with `Stock Movement Class` equal to 1.0 are labeled as "Moving," while others (e.g., 4.0) are labeled as "Non/Slow Moving."
*   **Categorical Feature Encoding**: The `type` column (indicating 'Supply', 'Fabric', or 'Spare Part') is one-hot encoded using `pd.get_dummies` to convert categorical data into a numerical format suitable for machine learning models.

### 3. Feature and Target Selection

The features selected for the models are:
*   `Unit_Price_in_TND`
*   `quantity`
*   One-hot encoded `type` columns (e.g., `type_Fabric`, `type_Spare Part`, `type_Supply`)

The target variable is `The Movement`.

### 4. Model Building and Evaluation

The dataset is split into training and testing sets (80% training, 20% testing) using `train_test_split` with a `random_state` for reproducibility.

Two classification models are implemented and evaluated:

#### a. Logistic Regression with Cross-Validation

*   **Model**: `LogisticRegressionCV` is used, which includes built-in cross-validation for hyperparameter tuning. This helps in selecting the optimal regularization parameter.
*   **Training**: The model is trained on the `X_train` and `y_train` datasets.
*   **Evaluation**: Predictions are made on the `X_test` set, and the model's performance is assessed using:
    *   **Accuracy Score**: Approximately 0.71.
    *   **Classification Report**: Provides precision, recall, and F1-score for both classes.
    *   **Confusion Matrix**: Visualized as a heatmap to show true positives, true negatives, false positives, and false negatives.

#### b. K-Nearest Neighbors (KNN) with Hyperparameter Tuning

*   **Model**: `KNeighborsClassifier` is employed.
*   **Hyperparameter Tuning**: `GridSearchCV` is utilized to find the optimal number of neighbors (`n_neighbors`), `weights` (uniform or distance), and `metric` (euclidean, manhattan, minkowski) for the KNN model. This ensures the model performs optimally.
*   **Pipeline**: A `Pipeline` is constructed to first scale the features using `StandardScaler` and then apply the tuned KNN classifier. This prevents features with larger scales from dominating the distance calculations.
*   **Cross-Validation**: `cross_val_score` is used to evaluate the pipeline's performance across multiple folds, providing a more robust estimate of its generalization ability.
*   **Training**: The pipeline is fitted on the entire training data.
*   **Evaluation**: Similar to Logistic Regression, predictions are made on `X_test`, and the model is evaluated using:
    *   **Accuracy Score**: Approximately 0.78.
    *   **Classification Report**: Provides precision, recall, and F1-score for both classes.
    *   **Confusion Matrix**: Visualized as a heatmap.

## Results

Both models demonstrate reasonable performance in classifying stock movement. The K-Nearest Neighbors model, after hyperparameter tuning and scaling, shows a slightly better accuracy of approximately 0.78 compared to Logistic Regression's 0.71. This suggests that the non-linear decision boundary and local neighborhood-based classification of KNN might be more suitable for this particular dataset.

## Conclusion and Future Work

This project successfully demonstrates the application of machine learning techniques to predict stock movement in the clothing industry. The KNN model, with its higher accuracy, provides a promising approach for identifying "Moving" and "Non/Slow Moving" items.

Future enhancements could include:
*   **More Advanced Feature Engineering**: Exploring additional features such as seasonality, supplier reliability, or product lifecycle stages.
*   **Other Classification Algorithms**: Experimenting with models like Support Vector Machines, Random Forests, or Gradient Boosting for potentially better performance.
*   **Time Series Analysis**: Incorporating time-series forecasting techniques to predict future stock levels and movements more dynamically.
*   **Deployment**: Developing a user-friendly interface for the model to be used by inventory managers.

## How to Run

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Install dependencies**:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
    ```
3.  **Place data files**: Ensure `Supply.xlsx`, `Fabric.xlsx`, and `Spare Part.xlsx` are in the same directory as the Jupyter notebook.
4.  **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook ClothingIndustry2022RawMaterialsandConsumablesDataScience.ipynb
    ```
    Follow the cells in the notebook to execute the code step-by-step.

## Dependencies

*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `matplotlib`
*   `seaborn`
*   `openpyxl` (for reading Excel files)
