import pandas as pd
import numpy as np
from impyute.imputation.cs import mice
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# =============================================================================
# Data Cleaning and Feature Standardization
# =============================================================================

# Identify numerical and categorical columns in the dataframe
def identify_column_types(df):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    return numerical_cols, categorical_cols

# Standardize categorical variables using predefined mappings
def standardize_features(df):
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
    item_type_mapping = {
        'Dairy': 'Food', 'Soft Drinks': 'Beverages', 'Meat': 'Food',
        'Fruits and Vegetables': 'Food', 'Household': 'Non-Consumable',
        'Baking Goods': 'Food', 'Snack Foods': 'Food', 'Frozen Foods': 'Food',
        'Breakfast': 'Food', 'Health and Hygiene': 'Non-Consumable',
        'Hard Drinks': 'Beverages', 'Canned': 'Food', 'Breads': 'Food',
        'Starchy Foods': 'Food', 'Others': 'Others', 'Seafood': 'Food'
    }
    df['Item_Type_Grouped'] = df['Item_Type'].map(item_type_mapping)
    return df



# =============================================================================
# Imputation for Item_Weight & Outlet_Size Using MICE 
# =============================================================================

# Impute missing Item_Weight values based on Item_Identifier
def fit_impute_item_weight(df):
    item_weights = df.groupby('Item_Identifier')['Item_Weight'].transform(lambda x: x.fillna(x.dropna().iloc[0]) if not x.dropna().empty else np.nan)
    df['Item_Weight'] = item_weights
    return df, item_weights

# Apply precomputed Item_Weight imputation
def transform_impute_item_weight(df, item_weights):
    df['Item_Weight'] = df['Item_Identifier'].map(item_weights).fillna(df['Item_Weight'])
    return df

# Apply MICE imputation for categorical features
def fit_mice_imputation(df, selected_columns, categorical_vars):
    work_df = df[selected_columns]
    mice_mappings = {}
    for col in categorical_vars:
        work_df.loc[:, col] = work_df[col].astype("category")
        mice_mappings[col] = dict(enumerate(work_df[col].cat.categories))
        work_df.loc[:, col] = work_df[col].cat.codes.replace(-1, np.nan)
    df_imputed = pd.DataFrame(mice(work_df.values), columns=work_df.columns)
    df['Outlet_Size'] = df_imputed['Outlet_Size'].round().astype(int).map(mice_mappings['Outlet_Size'])
    return df, mice_mappings

# Apply stored MICE mappings to test data
def transform_mice_imputation(df, selected_columns, categorical_vars, mice_mappings):
    work_df = df[selected_columns]
    for col in categorical_vars:
        work_df.loc[:, col] = work_df[col].astype("category")
        work_df.loc[:, col] = work_df[col].cat.codes.replace(-1, np.nan)
    df_imputed = pd.DataFrame(mice(work_df.values), columns=work_df.columns)
    df['Outlet_Size'] = df_imputed['Outlet_Size'].round().astype(int).map(mice_mappings['Outlet_Size'])
    return df



# =============================================================================
# Feature Engineering
# =============================================================================

# Create new derived features
def create_features(df):
    df['Item_Visibility'] = df['Item_Visibility'].replace(0, df['Item_Visibility'].median())
    df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']
    df['Establishment_Decade'] = (df['Outlet_Establishment_Year'] // 10) * 10
    df['Outlet_Age_Category'] = pd.cut(df['Outlet_Age'], bins=[0, 10, 20, 30, 40], labels=['New', 'Medium', 'Old', 'Very Old'])
    df['Price_per_Unit_Weight'] = df['Item_MRP'] / df['Item_Weight']
    df['Item_MRP_Binned'] = pd.qcut(df['Item_MRP'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    return df

# =============================================================================
# Outlier Detection and Removal
# =============================================================================

# Remove outliers using IQR method and test data constraints
def detect_and_remove_outliers(train_df, test_df, numerical_cols, threshold=1.5):
    for col in numerical_cols:
        Q1, Q3 = train_df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_limit, upper_limit = Q1 - threshold * IQR, Q3 + threshold * IQR
        train_df = train_df[(train_df[col] >= lower_limit) & (train_df[col] <= upper_limit)]
        if col in test_df.columns:
            test_min, test_max = test_df[col].min(), test_df[col].max()
            train_df = train_df[(train_df[col] >= test_min) & (train_df[col] <= test_max)]
    return train_df


# =============================================================================
# Data Transformation and Scaling
# =============================================================================

# Apply log transformations to features
def transform_features(df):
    if 'Item_Outlet_Sales' in df.columns:
        df['Item_Outlet_Sales'] = np.log1p(df['Item_Outlet_Sales'])
    df['Item_Visibility'] = np.log1p(df['Item_Visibility'])
    df['Price_per_Unit_Weight'] = np.log1p(df['Price_per_Unit_Weight'])
    return df

# Scale numerical features using RobustScaler
def fit_scale_numerical_features(df, num_cols):
    """
    Fit the scaler on training data, 'Item_Outlet_Sales' is excluded.
    """
    scaler = RobustScaler()
    filtered_cols = [col for col in num_cols if col != 'Item_Outlet_Sales']
    df[filtered_cols] = scaler.fit_transform(df[filtered_cols])
    return df, scaler

# Apply fitted scaler to test data
def transform_scale_numerical_features(df, num_cols, scaler):
    filtered_cols = [col for col in num_cols if col in df.columns]
    df[filtered_cols] = scaler.transform(df[filtered_cols])
    return df

# =============================================================================
# Categorical Feature Encoding
# =============================================================================

# Encode categorical features using ordinal encoding
def fit_encode_categorical_features(df):
    categorical_columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size',
                           'Outlet_Location_Type', 'Outlet_Type', 'Item_Type_Grouped',
                           'Outlet_Age_Category', 'Item_MRP_Binned']
    ordering_dict = {
        'Item_Fat_Content': ['Low Fat', 'Regular'],
        'Outlet_Size': ['Small', 'Medium', 'High'],
        'Outlet_Location_Type': ['Tier 3', 'Tier 2', 'Tier 1'],
        'Outlet_Type': ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'],
        'Item_Type_Grouped': ['Food', 'Beverages', 'Non-Consumable', 'Others'],
        'Outlet_Age_Category': ['New', 'Medium', 'Old', 'Very Old'],
        'Item_MRP_Binned': ['Low', 'Medium', 'High', 'Very High']
    }
    categories = [ordering_dict.get(col, sorted(df[col].unique())) for col in categorical_columns]
    encoder = OrdinalEncoder(categories=categories)
    df[categorical_columns] = encoder.fit_transform(df[categorical_columns]).astype(int)
    return df, encoder

# Apply fitted encoder to test data
def transform_encode_categorical_features(df, encoder):
    categorical_columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size',
                           'Outlet_Location_Type', 'Outlet_Type', 'Item_Type_Grouped',
                           'Outlet_Age_Category', 'Item_MRP_Binned']
    df[categorical_columns] = encoder.transform(df[categorical_columns]).astype(int)
    return df