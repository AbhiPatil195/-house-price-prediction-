# Data Description

This document provides information about the dataset used in this Kaggle project.

## Files

- `train_data.csv`: Training dataset with 20 examples
- `test_data.csv`: Test dataset with 20 examples
- `sample_submission.csv`: Example of the submission format

## Features

The dataset contains the following features:

1. `id`: Unique identifier for each example
2. `feature1`: A continuous numerical feature ranging from 0.5 to 2.4
3. `feature2`: A continuous numerical feature ranging from 7.5 to 13.1
4. `feature3`: A categorical feature with values A, B, or C
5. `target`: The binary target variable (0 or 1) to predict

## Task

This is a binary classification task. The goal is to predict the `target` variable (0 or 1) for the test set based on the features provided.

## Evaluation Metric

The evaluation metric for this competition is accuracy, which is the percentage of predictions that match the true target values.

## Data Insights

- Feature1 appears to have a positive correlation with the target
- Feature2 appears to have a negative correlation with the target
- Feature3 categories seem to have different distributions of target values:
  - Category A: More likely to be associated with target=0
  - Category B: Balanced distribution
  - Category C: More likely to be associated with target=1 