# Approach Note: BigMart Sales Prediction

## Problem Statement
The goal is to predict `Item_Outlet_Sales` for various items across different outlets using a regression model. The dataset includes features like item properties (e.g., weight, visibility, MRP) and outlet details (e.g., type, size, location). The evaluation metric is Root Mean Squared Error (RMSE).

## Data Preprocessing
1. **Missing Values**:
   - `Item_Weight`: Filled missing values with the mean weight per `Item_Type`.
   - `Outlet_Size`: Filled missing values with the mode per `Outlet_Type`, defaulting to 'Medium' if no mode exists.
   - `Item_Visibility`: Replaced zeros with the mean visibility per `Item_Identifier`, then filled remaining NaNs with the global mean.

2. **Feature Engineering**:
   - Created `Outlet_Age` by subtracting `Outlet_Establishment_Year` from 2013 (assumed competition year).
   - Standardized `Item_Fat_Content` by mapping variations (e.g., 'LF', 'low fat') to consistent categories ('Low Fat', 'Regular').

3. **Encoding**:
   - One-hot encoded categorical variables: `Item_Fat_Content`, `Item_Type`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`.
   - Dropped identifiers (`Item_Identifier`, `Outlet_Identifier`) and `Outlet_Establishment_Year` as they were redundant or transformed.

4. **Scaling**:
   - Applied `RobustScaler` to numerical features (`Item_Weight`, `Item_Visibility`, `Item_MRP`, `Outlet_Age`) to handle outliers effectively.

## Feature Selection
- Used Recursive Feature Elimination (RFE) with an `ExtraTreesRegressor` base model to select half of the features, reducing dimensionality and focusing on the most predictive ones.

## Modeling
1. **Hyperparameter Tuning**:
   - Employed Optuna with TPESampler to optimize hyperparameters for six regression models: RandomForest, XGBoost, LightGBM, CatBoost, ExtraTrees, and GradientBoosting.
   - Evaluated each model using 5-fold cross-validation and RMSE as the metric.
   - Selected the top 3 performing models based on RMSE.

2. **Stacking Ensemble**:
   - Trained the top 3 models on out-of-fold predictions using 5-fold cross-validation.
   - Used a `Ridge` regression meta-model to combine predictions from the base models.
   - Final predictions were generated for the test set using this stacked ensemble.

## Results
- Achieved a validation RMSE of **1144.85**, placing me at **135th** on the leaderboard.
- The stacking approach improved performance over individual models by leveraging their complementary strengths.

## Potential Improvements
1. **Feature Engineering**:
   - Explore interaction terms (e.g., `Item_MRP` × `Outlet_Type`).
   - Extract more features from `Item_Identifier` (e.g., category prefixes).
2. **Modeling**:
   - Test additional ensemble techniques (e.g., weighted averaging, blending).
   - Increase Optuna trials or refine hyperparameter ranges.
3. **Data Cleaning**:
   - Investigate outliers or anomalies in `Item_Visibility` and `Item_Outlet_Sales`.

## Files Included
- `EDA.ipynb`: Exploratory Data Analysis to understand data distributions and relationships.
- `modelling.py`: Full pipeline from preprocessing to final predictions.
- `submission.csv`: Final predictions for the test set.

## Conclusion
The solution combines robust preprocessing, feature selection, and a stacking ensemble to achieve a competitive RMSE. With further tuning and feature engineering, the model could climb higher on the leaderboard.
