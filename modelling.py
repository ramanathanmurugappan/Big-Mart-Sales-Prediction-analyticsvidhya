
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# Load training and testing datasets
train_data = pd.read_csv('train_v9rqX0R.csv')  
test_data = pd.read_csv('test_AbJTz2l.csv')    

# Filter out specific items from training data which are not in testing data
items_to_exclude = ['FDX20', 'FDG33', 'FDW13', 'FDG24', 'DRE49', 'NCY18', 'FDO19', 
                    'FDL34', 'FDO52', 'NCL31', 'FDA04', 'NCQ06', 'FDT07', 'FDL10', 
                    'FDX04', 'FDU19']
train_data = train_data[~train_data['Item_Identifier'].isin(items_to_exclude)]

# collate datasets for combined preprocessing 
train_data['dataset_type'] = 'train'
test_data['dataset_type'] = 'test'
full_data = pd.concat([train_data, test_data], ignore_index=True)

# Handle missing values and preprocess data
def preprocess_data(df):
    # Fill missing weights with mean per item type
    df['Item_Weight'] = df.groupby('Item_Type')['Item_Weight'].transform(lambda x: x.fillna(x.mean()))
    
    # Fill missing outlet sizes with mode per outlet type
    df['Outlet_Size'] = df.groupby('Outlet_Type')['Outlet_Size'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Medium'))
    
    # Replace zero visibility with item-specific mean, then overall mean
    visibility_means = df[df['Item_Visibility'] > 0].groupby('Item_Identifier')['Item_Visibility'].mean()
    df['Item_Visibility'] = df.apply(
        lambda row: visibility_means.get(row['Item_Identifier'], df['Item_Visibility'].mean()) 
        if row['Item_Visibility'] == 0 else row['Item_Visibility'], axis=1)
    
    # Compute outlet age
    df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']
    
    # Standardize fat content labels
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(
        {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})
    
    return df

full_data = preprocess_data(full_data)

# Encode categorical features
categorical_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 
                        'Outlet_Location_Type', 'Outlet_Type']
for feature in categorical_features:
    encoded_cols = pd.get_dummies(full_data[feature], prefix=feature)
    full_data = pd.concat([full_data, encoded_cols], axis=1)
    full_data.drop(feature, axis=1, inplace=True)

# Remove columns not needed for modeling
full_data.drop(['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'], 
               axis=1, inplace=True)

# Separate back into train and test sets
train_set = full_data[full_data['dataset_type'] == 'train'].drop('dataset_type', axis=1)
test_set = full_data[full_data['dataset_type'] == 'test'].drop(['dataset_type', 'Item_Outlet_Sales'], axis=1)

# Define features and target
features = train_set.drop('Item_Outlet_Sales', axis=1)
target = train_set['Item_Outlet_Sales']
test_features = test_set

# Scale numerical features
numeric_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years']
scaler = RobustScaler()
features[numeric_cols] = scaler.fit_transform(features[numeric_cols])
test_features[numeric_cols] = scaler.transform(test_features[numeric_cols])

# Feature selection using Recursive Feature Elimination
feature_selector_model = ExtraTreesRegressor(n_estimators=500, max_depth=10, 
                                             min_samples_leaf=35, n_jobs=-1, random_state=42)
n_selected_features = features.shape[1] // 2
rfe_selector = RFE(estimator=feature_selector_model, n_features_to_select=n_selected_features, step=1)
selected_features = rfe_selector.fit_transform(features, target)
test_selected_features = rfe_selector.transform(test_features)

# Split data for validation
X_train_subset, X_val_subset, y_train_subset, y_val_subset = train_test_split(
    selected_features, target, test_size=0.2, random_state=42)

# Optuna/Bayesian optimization function
def tune_model(trial):
    algo = trial.suggest_categorical('algorithm', ['rf', 'xgb', 'lgb', 'cat', 'et', 'gb'])
    
    if algo == 'rf':
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int('rf_trees', 100, 1000),
            max_depth=trial.suggest_int('rf_depth', 3, 15),
            min_samples_leaf=trial.suggest_int('rf_leaf', 10, 100),
            n_jobs=-1, random_state=42)
    elif algo == 'xgb':
        model = xgb.XGBRegressor(
            n_estimators=trial.suggest_int('xgb_trees', 100, 1000),
            max_depth=trial.suggest_int('xgb_depth', 3, 15),
            learning_rate=trial.suggest_float('xgb_lr', 0.01, 0.3),
            reg_alpha=trial.suggest_float('xgb_alpha', 0, 1),
            reg_lambda=trial.suggest_float('xgb_lambda', 0, 1),
            random_state=42)
    elif algo == 'lgb':
        model = lgb.LGBMRegressor(
            n_estimators=trial.suggest_int('lgb_trees', 100, 1000),
            max_depth=trial.suggest_int('lgb_depth', 3, 15),
            learning_rate=trial.suggest_float('lgb_lr', 0.01, 0.3),
            reg_alpha=trial.suggest_float('lgb_alpha', 0, 1),
            reg_lambda=trial.suggest_float('lgb_lambda', 0, 1),
            random_state=42, verbose=-1)
    elif algo == 'cat':
        model = cat.CatBoostRegressor(
            iterations=trial.suggest_int('cat_iters', 100, 1000),
            depth=trial.suggest_int('cat_depth', 3, 10),
            learning_rate=trial.suggest_float('cat_lr', 0.01, 0.3),
            l2_leaf_reg=trial.suggest_float('cat_l2', 1, 10),
            random_seed=42, verbose=0)
    elif algo == 'et':
        model = ExtraTreesRegressor(
            n_estimators=trial.suggest_int('et_trees', 100, 1000),
            max_depth=trial.suggest_int('et_depth', 3, 15),
            min_samples_leaf=trial.suggest_int('et_leaf', 10, 100),
            n_jobs=-1, random_state=42)
    elif algo == 'gb':
        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int('gb_trees', 100, 1000),
            max_depth=trial.suggest_int('gb_depth', 3, 15),
            learning_rate=trial.suggest_float('gb_lr', 0.01, 0.3),
            min_samples_leaf=trial.suggest_int('gb_leaf', 10, 100),
            random_state=42)
    
    # Cross-validation
    cv_folds = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list = []
    for train_fold, val_fold in cv_folds.split(features):
        X_fold_train, X_fold_val = features.iloc[train_fold], features.iloc[val_fold]
        y_fold_train, y_fold_val = target.iloc[train_fold], target.iloc[val_fold]
        model.fit(X_fold_train, y_fold_train)
        preds = model.predict(X_fold_val)
        rmse_list.append(np.sqrt(mean_squared_error(y_fold_val, preds)))
    
    trial.set_user_attr('hyperparams', {k: v for k, v in trial.params.items() if k != 'algorithm'})
    return np.mean(rmse_list)

# Run hyperparameter tuning
tpe_sampler = TPESampler(seed=42)
optimization_study = optuna.create_study(direction='minimize', sampler=tpe_sampler)
optimization_study.optimize(tune_model, n_trials=50)

# Select top 3 performing models
sorted_trials = sorted([t for t in optimization_study.trials if t.value is not None], key=lambda x: x.value)
best_trials = sorted_trials[:3]
print("Best 3 models:", [(t.params['algorithm'], t.value) for t in best_trials])

# Build selected models
selected_models = []
for trial in best_trials:
    algo = trial.params['algorithm']
    params = {k.split('_', 1)[1] if '_' in k else k: v for k, v in trial.params.items() if k != 'algorithm'}
    if algo == 'rf':
        model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
    elif algo == 'xgb':
        model = xgb.XGBRegressor(**params, random_state=42)
    elif algo == 'lgb':
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
    elif algo == 'cat':
        if 'iters' in params:
            params['iterations'] = params.pop('iters')
        model = cat.CatBoostRegressor(**params, random_seed=42, verbose=0)
    elif algo == 'et':
        model = ExtraTreesRegressor(**params, n_jobs=-1, random_state=42)
    elif algo == 'gb':
        model = GradientBoostingRegressor(**params, random_state=42)
    selected_models.append((algo, model))

# Perform stacking ensemble
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
out_of_fold_preds = {f'{algo}_{idx}': np.zeros(len(selected_features)) 
                     for idx, (algo, _) in enumerate(selected_models)}
test_predictions = {f'{algo}_{idx}': np.zeros(len(test_selected_features)) 
                    for idx, (algo, _) in enumerate(selected_models)}

for train_indices, val_indices in cv_strategy.split(selected_features):
    X_tr_fold, X_val_fold = selected_features[train_indices], selected_features[val_indices]
    y_tr_fold, y_val_fold = target.iloc[train_indices], target.iloc[val_indices]
    for idx, (algo, model) in enumerate(selected_models):
        model.fit(X_tr_fold, y_tr_fold)
        out_of_fold_preds[f'{algo}_{idx}'][val_indices] = model.predict(X_val_fold)
        test_predictions[f'{algo}_{idx}'] += model.predict(test_selected_features) / cv_strategy.n_splits

# Train meta-model
meta_X = np.column_stack([out_of_fold_preds[key] for key in out_of_fold_preds])
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(meta_X, target)
final_test_preds = meta_learner.predict(np.column_stack([test_predictions[key] for key in test_predictions]))

# Create submission file
submission_df = pd.DataFrame({
    'Item_Identifier': test_data['Item_Identifier'],
    'Outlet_Identifier': test_data['Outlet_Identifier'],
    'Item_Outlet_Sales': final_test_preds
})
submission_df.to_csv('final_submission.csv', index=False)
print("Submission saved as 'final_submission.csv'")
