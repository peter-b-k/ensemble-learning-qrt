# Ensemble Learning Project: Explanation for the Electricity Price 
## Description
This project aims to model the electricity price from weather, energy and commercial data for two European  countries- France and Germany. Our object is to use several models to find out the best explanation for the price, and our evaluation method will be MSE, MAE and Spearman correlation. The models we want to compare are as follows: 
- **Decision Trees** 
- **Randoms Forests**
- **Bagging** 
- **AdaBoost**
- **Gradient Boost**
- **Extra Tree** 
- **XGB**

## Dependencies
- **pandas** 
- **numpy**
- **matplotlib** 
- **sklearn**
- **xgboost**

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is an essential step in the data analysis process, where the main goal is to understand the structure and characteristics of the dataset. During EDA, various statistical and visual techniques are employed to uncover patterns, detect anomalies, and gain insights into the data.

### Key Components of EDA:

1. **Summary Statistics**: Calculate and review summary statistics such as mean, median, mode, standard deviation, etc., to get an overall understanding of the dataset.

2. **Visualization**: Utilize various types of plots and charts to visualize the distributions and relationships among variables.

3. **Missing Values**: Identify and handle missing values appropriately, as they can affect the analysis and modeling process.

4. **Outlier Detection**: Detect and analyze outliers, which are data points that significantly differ from other observations in the dataset.

5. **Correlation Analysis**: Examine the relationships between variables using correlation matrices or correlation coefficients.

 In this project, the columns can be grouped by four parts: Daily Commodity Price Variation, Weather Measures, Energy Production Measures and Electricity Use Metrics.
  
We observe the distribution characteristics of the four categories of columns through histograms, as well as the differences in distribution between the two countries. We find that there are no significant country differences in the "Daily Commodity Price Variation" and "Weather Measures" feature groups. However, the "Energy Production Measures" and "Electricity Use Metrics" exhibit distinct features between countries. Therefore, it is necessary to consider the two countries separately when modeling.

## Data Preprocessing
The goal of data preprocessing is to clean, transform, and prepare the dataset for analysis and modeling. In this project, the data preprocessing steps involve loading the data, splitting it into separate datasets for France (FR) and Germany (DE), performing KNN imputation to handle missing values, and trimming extreme values from certain features.

Here's a brief summary of the preprocessing steps implemented in the code:

1. **Loading Data**: The dataset is loaded from CSV files containing features and target variables. Both training and testing datasets are loaded separately.

2. **Separating Countries**: If specified, the data is split into two separate datasets for France (FR) and Germany (DE). This separation allows for individual preprocessing and modeling for each country.

3. **KNN Imputation**: Missing values in the dataset are imputed using the K-Nearest Neighbors (KNN) imputation method. This technique replaces missing values with estimated values based on the nearest neighbors' information.

4. **Trimming Tail**: To reduce the influence of extreme values, the tails of certain features are trimmed. This is done by clipping the values to a specified percentile range, effectively removing outliers from the dataset.

5. **One-Hot Encoding**: If not separating countries, one-hot encoding is applied to categorical features (e.g., country codes) to convert them into numerical format, making them suitable for modeling.

6. **Feature Engineering**: Additional features may be derived or transformed based on domain knowledge or specific requirements to enhance the predictive power of the models.

By following these preprocessing steps, the data is cleaned, imputed, and transformed into a suitable format for further analysis and modeling. This ensures that the models trained on the data are robust and capable of making accurate predictions.

## Feature Engineering
Feature engineering involves creating new features or transforming existing ones to improve the predictive performance of machine learning models. Here are the key feature engineering techniques implemented in the project:

### Lagged Items
Lagged items are created separately for Germany (DE) and France (FR). In-week lag items are built for selected features to capture temporal dependencies in the data.

### Consumption Inspirations
Consumption inspirations are derived from various aspects related to energy consumption and production. This includes:

1. **Average Commodity Price Variations**: The average variations in commodity prices (e.g., gas, coal, carbon) are calculated and smoothed using moving averages.

2. **Nuclear Ratio Trend**: Trends in the ratio of nuclear energy production to total energy production are captured for both DE and FR.

3. **New Energy Transformation Efficiency**: Efficiency measures for hydro and wind energy production relative to environmental factors (e.g., rainfall, wind speed) are computed.

4. **Residual Load Premium Cost**: The cost implications of residual load and net imports are estimated based on commodity price variations.

After creating new features, all of them are standardized to ensure uniformity in their scale. Standardization is performed using the StandardScaler from scikit-learn to make features comparable and improve model performance.

By incorporating these feature engineering techniques, the dataset is enriched with meaningful features that enhance the predictive power of the machine learning models.

## Modeling Toolkit

The modeling toolkit consists of utility functions and a modeling pipeline designed to streamline the process of building and evaluating regression models. Here's an overview of the key components:

### Utility Functions

- **`metric_kit(y_test, y_pred)`:** Calculates and displays metrics for model evaluation, including Mean Squared Error (MSE), Mean Absolute Error (MAE), and Spearman correlation coefficient.

### Modeling Pipeline

- **`modeling_pipeline(X_train, Y_train, model_regressor, param_grid)`:** Performs the modeling pipeline, which includes:
  - Splitting the dataset into training and validation sets.
  - Defining the regression model.
  - Performing hyperparameter tuning using GridSearchCV.
  - Selecting the best model based on hyperparameter tuning results.
  - Evaluating the best model on the validation set using the `metric_kit` function.
  
The modeling toolkit provides a convenient way to train and evaluate regression models, helping streamline the model development process.

## Model Comparison

#### Decision Tree
- **Description**: Decision Tree model is a non-parametric supervised learning method used for classification and regression tasks. It creates a tree-like structure where each internal node represents a feature, each branch represents a decision based on that feature, and each leaf node represents the outcome.
- **Best Parameters (FR)**:
  - Criterion: 'absolute_error'
  - Max Depth: 10
  - Min Samples Leaf: 4
  - Min Samples Split: 10
- **Metrics (FR)**:
  - MSE: 0.981
  - MAE: 0.636
  - Spearman Correlation: 43.5%
- **Best Parameters (DE)**:
  - Criterion: 'absolute_error'
  - Max Depth: 10
  - Min Samples Leaf: 2
  - Min Samples Split: 10
- **Metrics (DE)**:
  - MSE: 1.115
  - MAE: 0.551
  - Spearman Correlation: 21.7%

#### Random Forest
- **Description**: Random Forest is an ensemble learning method that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
- **Best Parameters (FR)**:
  - n_estimators: 100
  - Max Depth: 15
  - Min Samples Leaf: 4
  - Min Samples Split: 2
- **Metrics (FR)**:
  - MSE: 0.988
  - MAE: 0.518
  - Spearman Correlation: 7.0%
- **Best Parameters (DE)**:
  - n_estimators: 100
  - Max Depth: None
  - Min Samples Leaf: 4
  - Min Samples Split: 2
- **Metrics (DE)**:
  - MSE: 0.535
  - MAE: 0.478
  - Spearman Correlation: 57.4%

#### Bagging
- **Description**: Bagging is an ensemble meta-algorithm that fits multiple base models on different subsets of the dataset and then combines their individual predictions to form a final prediction.
- **Best Parameters (FR)**:
  - n_estimators: 200
  - Max Samples: 0.5
  - Max Features: 0.5
- **Metrics (FR)**:
  - MSE: 0.987
  - MAE: 0.510
  - Spearman Correlation: 13.1%
- **Best Parameters (DE)**:
  - n_estimators: 100
  - Max Samples: 0.5
  - Max Features: 1.0
- **Metrics (DE)**:
  - MSE: 0.556
  - MAE: 0.495
  - Spearman Correlation: 54.4%

#### AdaBoost
- **Description**: AdaBoost is a boosting technique that builds multiple weak learners sequentially. Each new model attempts to correct the errors made by the previous ones, with more weight placed on difficult-to-classify instances.
- **Best Parameters (FR)**:
  - n_estimators: 100
  - Learning Rate: 0.1
- **Metrics (FR)**:
  - MSE: 0.999
  - MAE: 0.487
  - Spearman Correlation: 1.0%
- **Best Parameters (DE)**:
  - n_estimators: 100
  - Learning Rate: 0.1
- **Metrics (DE)**:
  - MSE: 0.626
  - MAE: 0.545
  - Spearman Correlation: 55.0%

#### Gradient Boost
- **Description**: Gradient Boosting builds an additive model in a forward stage-wise manner, where subsequent models fit the residual errors of the previous models.
- **Best Parameters (FR)**:
  - n_estimators: 50
  - Learning Rate: 0.01
- **Metrics (FR)**:
  - MSE: 0.970
  - MAE: 0.461
  - Spearman Correlation: -2.3%
- **Best Parameters (DE)**:
  - n_estimators: 50
  - Learning Rate: 0.05
- **Metrics (DE)**:
  - MSE: 0.582
  - MAE: 0.501
  - Spearman Correlation: 55.3%

#### Extra Trees
- **Description**: Extra Trees is another ensemble method similar to Random Forest but differs in the way it selects the splitting thresholds.
- **Best Parameters (FR)**:
  - n_estimators: 100
  - Max Depth: 20
- **Metrics (FR)**:
  - MSE: 1.030
  - MAE: 0.558
  - Spearman Correlation: 6.4%
-

 **Best Parameters (DE)**:
  - n_estimators: 500
  - Max Depth: None
- **Metrics (DE)**:
  - MSE: 0.542
  - MAE: 0.475
  - Spearman Correlation: 59.5%

#### XGBoost
- **Description**: XGBoost (Extreme Gradient Boosting) is an implementation of gradient boosting that is highly efficient and flexible.
- **Best Parameters (FR)**:
  - n_estimators: 50
  - Learning Rate: 0.01
  - Max Depth: 3
- **Metrics (FR)**:
  - MSE: 0.974
  - MAE: 0.465
  - Spearman Correlation: -5.8%
- **Best Parameters (DE)**:
  - n_estimators: 150
  - Learning Rate: 0.01
  - Max Depth: 3
- **Metrics (DE)**:
  - MSE: 0.537
  - MAE: 0.474
  - Spearman Correlation: 58.4%


### Model Comparison Summary

We can also directly observe the evaluation of each model through these two graphs:
![WhatsApp 图像2024-03-03于21 51 22_7ffcff8d](https://github.com/peter-b-k/ensemble-learning-qrt/assets/156606885/9320fe37-af31-45be-875b-526842cf48a1)
![WhatsApp 图像2024-03-03于21 51 33_13ec4b26](https://github.com/peter-b-k/ensemble-learning-qrt/assets/156606885/115ba3f2-a98f-440b-ad55-fbc4a94463d3)

Based on the results, we can observe that Random Forest and XGBoost generally perform better compared to other models for both FR and DE datasets. They exhibit lower Mean Squared Error (MSE) and Mean Absolute Error (MAE) and higher Spearman Correlation, indicating better predictive performance. Additionally, Random Forest and XGBoost models demonstrate relatively stable performance across different datasets and parameter configurations. However, the choice of the best model may depend on specific requirements and constraints of the problem domain.

















