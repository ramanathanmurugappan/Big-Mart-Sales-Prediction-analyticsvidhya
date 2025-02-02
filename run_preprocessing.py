import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore")
from preprocessing_utils import (
    identify_column_types,
    standardize_features,
    fit_impute_item_weight,
    transform_impute_item_weight,
    fit_mice_imputation,
    transform_mice_imputation,
    create_features,
    detect_and_remove_outliers,
    transform_features,
    fit_scale_numerical_features,
    transform_scale_numerical_features,
    fit_encode_categorical_features,
    transform_encode_categorical_features
)


print("Loading datasets...")
train_df = pd.read_csv('train_v9rqX0R.csv')
test_df = pd.read_csv('test_AbJTz2l.csv')

print("Processing training set...")
# Step 1: Initial preprocessing
numerical_cols, categorical_cols = identify_column_types(train_df)
train_df = standardize_features(train_df)

# Step 2: Weight imputation
train_df, item_weights = fit_impute_item_weight(train_df)

# Step 3: MICE imputation
selected_columns = [
    "Item_Identifier", "Item_Weight", "Item_Fat_Content", "Item_Visibility",
    "Item_Type", "Item_MRP", "Outlet_Identifier", "Outlet_Establishment_Year",
    "Outlet_Size", "Outlet_Location_Type", "Outlet_Type","Item_Type_Grouped"]
categorical_vars = selected_columns
train_df, mice_mappings = fit_mice_imputation(train_df, selected_columns, categorical_vars)

# Step 4: Feature engineering
train_df = create_features(train_df)
train_df = transform_features(train_df)

# Step 5: Outlier detection and removal
numerical_cols, categorical_cols= identify_column_types(train_df)
train_df = detect_and_remove_outliers(train_df, test_df, numerical_cols)

# Step 6: Scaling
numerical_cols, categorical_cols= identify_column_types(train_df)
train_df, scaler = fit_scale_numerical_features(train_df, numerical_cols)

# Step 7: Encoding
train_df, encoder = fit_encode_categorical_features(train_df)

print("\nProcessing test set...")
# Process test set using fitted transformations but without outlier removal
test_df = standardize_features(test_df)
test_df = transform_impute_item_weight(test_df, item_weights)
numerical_cols, categorical_cols = identify_column_types(test_df)
test_df = transform_mice_imputation(test_df, selected_columns, categorical_vars, mice_mappings)
test_df = create_features(test_df)
test_df = transform_features(test_df)
numerical_cols, categorical_cols = identify_column_types(test_df)
test_df = transform_scale_numerical_features(test_df, numerical_cols, scaler)
test_df = transform_encode_categorical_features(test_df, encoder)

print("\nSaving processed datasets...")
train_df.to_csv('processed_train.csv', index=False)
test_df.to_csv('processed_test.csv', index=False)
print("Done! Processed datasets saved as 'processed_train.csv' and 'processed_test.csv'")

