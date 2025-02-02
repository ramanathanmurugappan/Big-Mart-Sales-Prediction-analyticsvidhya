import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import root_mean_squared_error


# =============================================================================
# Model Training and Evaluation
# =============================================================================

training_data = pd.read_csv('processed_train.csv')
testing_data = pd.read_csv('processed_test.csv')


# Split dataset into training and testing sets
X_test = training_data['Item_Outlet_Sales']
X_train = training_data.drop(columns=[col for col in ['Item_Outlet_Sales', 'Item_Identifier'] if col in training_data], errors='ignore')
y_train = training_data['Item_Outlet_Sales'] if 'Item_Outlet_Sales' in training_data else None

# Best CatBoost parameters from TPESampler
best_params = {
    "iterations": 296,
    "depth": 7,
    "learning_rate": 0.021573131062485866,
    "l2_leaf_reg": 4
}

# Initialize and train the CatBoost model
catboost_model = CatBoostRegressor(**best_params, verbose=0)
catboost_model.fit(X_train, y_train)

# Make predictions
y_pred = catboost_model.predict(X_test)

# Reverse log transformation (convert back to original scale)
y_pred = np.expm1(y_pred)

# Create submission dataframe
submission = pd.DataFrame({
    'Item_Identifier': testing_data['Item_Identifier'],
    'Outlet_Identifier': testing_data['Outlet_Identifier'],
    'Item_Outlet_Sales': y_pred
})

# Save predictions to submission file
submission.to_csv('submission.csv', index=False)
