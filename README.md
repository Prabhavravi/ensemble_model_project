**Ensemble Model: Predictive Regression Project**

**Project Overview**

This repository contains my first end-to-end machine learning project. The objective was to build a robust predictive model using an ensemble approach to estimate a continuous target variable. By combining the strengths of multiple gradient-boosting algorithms, I aimed to maximize predictive accuracy and generalization.

This project was built on an anonymized industry dataset. Without domain context, the focus was entirely on rigorous methodology.Then let the Nelder-Mead optimization, correlation-based feature reduction, and ensemble strategy carry the weight. Make the methodology section so detailed that the dataset almost doesn't matter.

**Methodology**

The project follows a complete data science pipeline:

.Preprocessing: Handled categorical variables using ordinal and frequency encoding to prepare features for modeling.

.Data Cleaning: Conducted Pearson correlation analysis to identify and remove highly correlated (redundant) features, effectively reducing noise.

.Imputation: Applied median imputation to handle missing values, ensuring a complete dataset for training.

.Feature Engineering: Log-transformed the target variable to stabilize variance and improve model performance.

.Ensemble Strategy: Instead of standard averaging, I implemented Nelder-Mead optimization to mathematically derive the optimal weights for my XGBoost, LightGBM, and CatBoost models.

.Validation: Used 5-Fold Cross-Validation to ensure the model remains robust and performs consistently across different data splits.

**Tools & Libraries**

.Language: Python

.Data Manipulation: pandas, numpy

.Machine Learning: scikit-learn, xgboost, lightgbm, catboost

.Optimization: scipy.optimize

**Performance**
The final model leverages a weighted ensemble of the following algorithms:

.XGBoost Regressor

.LightGBM Regressor

.CatBoost Regressor

By assigning weights based on the Nelder-Mead method, the final prediction provides a balanced and robust output that outperforms individual models.

**How to Run**
Run this command to install the necessary libraries
pip install -r requirements.txt
