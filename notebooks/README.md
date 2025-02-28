# House Price Prediction - Ames Housing Dataset

This directory contains code for predicting house prices using the Ames Housing dataset from Kaggle.

## Files

- `house_price_prediction.py`: Main Python script for data preprocessing, model training, and prediction

## How to Run

1. Install the required packages:
   ```
   pip install -r ../requirements.txt
   ```

2. Run the script:
   ```
   python house_price_prediction.py
   ```

3. The script will:
   - Load and preprocess the data
   - Train multiple regression models
   - Evaluate each model using cross-validation
   - Select the best model
   - Make predictions on the test set
   - Save the predictions to a submission file

## Model Details

The script trains and evaluates several regression models:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Elastic Net
5. Random Forest
6. Gradient Boosting
7. XGBoost

The best performing model will be used to make the final predictions.

## Feature Engineering

The script performs several feature engineering steps:

- Handling missing values
- Creating new features (e.g., total square footage, total bathrooms)
- Encoding categorical variables
- Transforming skewed numerical features
- One-hot encoding categorical features

## Output

The script will generate a submission file in the `../sample_submission/` directory named `my_submission.csv`, which can be submitted to the Kaggle competition. 