# Sales Prediction Model using Machine Learning

## Project Overview
This project aims to build a machine learning model to predict sales for products in retail stores. The dataset includes various features like product attributes and store characteristics. The goal is to leverage regression models to predict the `Item_Outlet_Sales` for each product.

## Models Used
The following regression models have been implemented and evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Support Vector Regressor (SVR)

## Features
- Data Preprocessing: Handling missing values, feature scaling, and feature selection.
- Model Evaluation: Evaluation based on metrics such as R², RMSE, and MAE.
- Hyperparameter Tuning: Using `GridSearchCV` for tuning Random Forest Regressor to improve model performance.
- Cross-Validation: Employed cross-validation for robust model evaluation.

## Libraries Used
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Performance Metrics
The following metrics were used to evaluate the performance of the models:
- **R²** (Coefficient of Determination)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

## Best Model
After performing hyperparameter tuning, the best-performing model was the **Random Forest Regressor**, which provided the highest R² score.
