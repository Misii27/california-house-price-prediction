# California-house-price-prediction
Linear Regression model predicting California house prices using Kaggle housing data
### Problem statement
UrbanNest Properties was estimating house prices manually and that led to inconsistent valuations, revenue loss and lack of trust from customers. This project builds a machine learning model that predicts median house prices, giving the business a consistent, data-backed pricing baseline.

**Dataset:** California Housing Dataset (Kaggle) - 20,640 records, 10 features including location, income, housing age, and ocean proximity.

## Project Workflow
* Data loading and inspection
* Missing value handling
* Exploratory Data Analysis (EDA)
* Outlier treatment
* Feature engineering and encoding
* Model training and evaluation

## Key Findings from EDA
* Median income is the strongest predictor of house price (correlation: 0.69). Neighbourhoods with higher incomes consistently have higher property values.
* Ocean proximity drives price. Island and near-bay properties command the highest average values while inland properties are significantly lower.
* Total rooms, total bedrooms, and households are highly correlated with each other (0.93–0.97), indicating multicollinearity.
* House price distribution is right-skewed. Most homes fall in the low-to-mid range, with fewer high-value properties pulling the tail.
* Outliers exist in total rooms, bedrooms, population, and households. IQR capping was applied to the upper bound only, since box plots showed no significant lower-end outliers.

## Data Preprocessing Decisions
* Missing values: 207 missing rows was filled with median
* Outlier treatment: To preserve the data volume, capping was done. IQR capping was applied to the upper bound only
* Categorical encoding: Ocean_proximity was one-hot encoded, creating dummy columns and INLAND was dropped to avoid multicollinearity
* MinMaxScaler: Made sure no single feature dominates due to scale differences
* Dropped features: longitude & latitude had no effect since ocean proximity already encode location

## Model
Algorithm: Linear Regression
Train/Test Split: 70% train, 30% test
## Results
The model explains about 64% of the variation in house prices across the test set. On average, predictions are off by roughly $50,000 from the actual value. The RMSE being higher than the MAE ($67K vs $50K) indicates that the model struggles more on high-value properties.

## Tools and Libraries
Python, pandas, NumPy, Matplotlib, Seaborn (visualisation), scikit-learn (preprocessing, modelling, evaluation), Jupyter notebook
