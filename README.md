# Explanation for the electricity price 
## Description
This project aims to model the electricity price from weather, energy and commercial data for two European  countries- France and Germany. Our object is to use several models to find out the best explanation for the price, and our evaluation method will be MSE, MAE and Spearman correlation. The models we want to compare are as follows: 
· Decision Trees
· Randoms Forests
· Bagging
· AdaBoost
· Gradient Boost
· Extra Tree
· XGB

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

## Featur Engineering
Feature engineering involves creating new features or transforming existing ones to improve the predictive performance of machine learning models. Here are the key feature engineering techniques implemented in the project:

### Lagged Items
Lagged items are created separately for Germany (DE) and France (FR). In-week lag items are built for selected features to capture temporal dependencies in the data.

### Consumption Inspirations
Consumption inspirations are derived from various aspects related to energy consumption and production. This includes:

1.**Average Commodity Price Variations**: The average variations in commodity prices (e.g., gas, coal, carbon) are calculated and smoothed using moving averages.

2.**Nuclear Ratio Trend**: Trends in the ratio of nuclear energy production to total energy production are captured for both DE and FR.

3.**New Energy Transformation Efficiency**: Efficiency measures for hydro and wind energy production relative to environmental factors (e.g., rainfall, wind speed) are computed.

4.**Residual Load Premium Cost**: The cost implications of residual load and net imports are estimated based on commodity price variations.

After creating new features, all of them are standardized to ensure uniformity in their scale. Standardization is performed using the StandardScaler from scikit-learn to make features comparable and improve model performance.

By incorporating these feature engineering techniques, the dataset is enriched with meaningful features that enhance the predictive power of the machine learning models.





















