#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction - Ames Housing Dataset
# 
# This script performs house price prediction using the Ames Housing dataset from Kaggle.
# It includes data exploration, preprocessing, feature engineering, model training, and prediction.
# This version does not require XGBoost.

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

# Set paths to data files
train_path = '../train/train.csv'
test_path = '../test/test.csv'
submission_path = '../sample_submission/sample_submission.csv'

# Load the data
print("Loading data...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(submission_path)

# Display basic information
print(f"Train data shape: {train.shape}")
print(f"Test data shape: {test.shape}")

# Save the 'Id' column for submission
train_ID = train['Id']
test_ID = test['Id']

# Remove the 'Id' column from the training and test sets
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

# Data Exploration
print("\n--- Data Exploration ---")

# Check for missing values
print("\nMissing values in training data:")
missing_train = train.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
print(missing_train.head(10))

print("\nMissing values in test data:")
missing_test = test.isnull().sum()
missing_test = missing_test[missing_test > 0].sort_values(ascending=False)
print(missing_test.head(10))

# Target variable analysis
print("\nTarget variable (SalePrice) analysis:")
print(f"Min: {train['SalePrice'].min()}")
print(f"Max: {train['SalePrice'].max()}")
print(f"Mean: {train['SalePrice'].mean()}")
print(f"Median: {train['SalePrice'].median()}")
print(f"Skewness: {train['SalePrice'].skew()}")
print(f"Kurtosis: {train['SalePrice'].kurt()}")

# Log transform the target variable to make it more normally distributed
train["SalePrice"] = np.log1p(train["SalePrice"])
print(f"Skewness after log transformation: {train['SalePrice'].skew()}")

# Combine train and test data for preprocessing
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train.drop('SalePrice', axis=1), test)).reset_index(drop=True)

print(f"Combined data shape: {all_data.shape}")

# Data Preprocessing
print("\n--- Data Preprocessing ---")

# Fill missing values
print("Filling missing values...")

# For numerical features with NA meaning something (e.g., no basement)
for col in ('GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

# For categorical features, NA often means "None"
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'MasVnrType', 'MSSubClass', 'PoolQC', 'Fence', 'MiscFeature', 'Alley'):
    all_data[col] = all_data[col].fillna('None')

# Fill LotFrontage based on the median of the neighborhood
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))

# Fill remaining missing values with mode
for col in all_data.columns:
    if all_data[col].isnull().sum() > 0:
        if all_data[col].dtype == np.object:  # Categorical
            all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
        else:  # Numerical
            all_data[col] = all_data[col].fillna(all_data[col].median())

# Verify no missing values remain
assert all_data.isnull().sum().sum() == 0, "There are still missing values in the data!"

# Feature Engineering
print("\n--- Feature Engineering ---")

# Convert some numerical features to categorical
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

# Create new features
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBathrooms'] = all_data['FullBath'] + (0.5 * all_data['HalfBath']) + \
                             all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath'])
all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['Has2ndFloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasFireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
all_data['Remodeled'] = (all_data['YearRemodAdd'] != all_data['YearBuilt']).astype(int)
all_data['HouseAge'] = 2010 - all_data['YearBuilt']
all_data['RemodAge'] = 2010 - all_data['YearRemodAdd']

# Label encode categorical features
print("Encoding categorical features...")
categorical_features = all_data.select_dtypes(include=['object']).columns
for col in categorical_features:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col].astype(str))

# Handle skewed numerical features
print("Transforming skewed numerical features...")
numeric_features = all_data.select_dtypes(include=['int64', 'float64']).columns
skewed_features = all_data[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed_features = skewed_features[abs(skewed_features) > 0.75]
print(f"Number of skewed features: {len(skewed_features)}")

for feat in skewed_features.index:
    all_data[feat] = np.log1p(all_data[feat])

# Get dummy variables for categorical features
all_data = pd.get_dummies(all_data)
print(f"Shape after getting dummies: {all_data.shape}")

# Recreate train and test sets
X_train = all_data[:ntrain]
X_test = all_data[ntrain:]

# Model Training
print("\n--- Model Training ---")

# Define evaluation function
def rmse_cv(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse

# Define models (without XGBoost)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=10),
    "Lasso": make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1)),
    "Elastic Net": make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3)),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=1500, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, random_state=1)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    score = rmse_cv(model, X_train, y_train)
    results[name] = score
    print(f"{name} RMSE: {score.mean():.4f} ({score.std():.4f})")

# Find the best model
best_model_name = min(results, key=lambda x: results[x].mean())
print(f"\nBest model: {best_model_name} with RMSE: {results[best_model_name].mean():.4f}")

# Train the best model on the full training data
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# Make predictions
print("\n--- Making Predictions ---")
predictions = best_model.predict(X_test)

# Transform predictions back from log scale
predictions = np.expm1(predictions)

# Create submission file
submission = pd.DataFrame({
    'Id': test_ID,
    'SalePrice': predictions
})
submission_file = '../sample_submission/my_submission.csv'
submission.to_csv(submission_file, index=False)
print(f"Submission file saved to {submission_file}")

# Feature importance (if applicable)
if best_model_name in ["Random Forest", "Gradient Boosting"]:
    importance = best_model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))

print("\nDone!") 