**Approach Note: Big Mart Sales Prediction**

### Objective
The goal of this project was to develop a robust machine learning model to predict sales for various products across different retail outlets. The approach included a structured pipeline of exploratory data analysis (EDA), preprocessing, feature engineering, model experimentation, and optimization.

### Exploratory Data Analysis (EDA)
1. **Data Structure Analysis**: Identified numerical and categorical features- Automated.
2. **Data Cleaning**: Standardized categorical values in `Item_Fat_Content`, `Outlet_Size`, handled missing values in `Item_Weight` and `Outlet_Size` using Multiple Imputation by Chained Equations (MICE), and imputed zero values in `Item_Visibility` with median.
3. **Feature Analysis**: Examined sales distribution (`Item_Outlet_Sales`), outlet characteristics (`Outlet_Type`, `Outlet_Location_Type`), and pricing trends (`Item_MRP`).
4. **Outlier Detection**: Used IQR-based filtering to remove anomalies in `Item_Weight` and `Item_Visibility` while ensuring test data validity Since its Skewed.

### Preprocessing & Feature Engineering
1. **Categorical Encoding**: Used ordinal encoding for ordered categories (`Outlet_Size`, `Outlet_Location_Type`,`Outlet_Type`, `Item_Type_Grouped`).
2. **Feature Creation**: Introduced `Outlet_Age` (derived from `Outlet_Establishment_Year`), `Price_per_Unit_Weight` (`Item_MRP` / `Item_Weight`), and binned `Item_MRP_Binned` categories to improve predictive power.
3. **Scaling**: Applied `RobustScaler` to handle skewed distributions in `Item_Visibility`, `Price_per_Unit_Weight`, and `Item_Outlet_Sales` effectively.
4. **Transformation**: Log-transformed skewed features (`Item_Visibility`, `Item_Outlet_Sales`, `Price_per_Unit_Weight`) to enhance model performance.

### Model Experimentation
1. **Baseline Models**: Initially experimented with `RandomForestRegressor`, `XGBoost`, `LightGBM`, and `CatBoost` using `Item_MRP`, `Item_Visibility`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Age`, `Item_Fat_Content`, and `Item_Type_Grouped` as features.
2. **Performance Evaluation**: Used `RMSE` as the evaluation metric and validated models on a 70-30 train-test split.
3. **Best Model Selection**: `CatBoost` achieved the best `RMSE` (~0.5206), making it the final choice for prediction.
4. **Hyperparameter Tuning**: Leveraged `Optuna`’s `TPE Sampler` to fine-tune `CatBoost` parameters (`iterations`, `depth`, `learning_rate`), improving its generalization ability.

### Results & Final Submission
- After hyperparameter tuning, the final model achieved the lowest `RMSE`.
- The trained model was used to generate predictions on the test set, with log-transformed values (`Item_Outlet_Sales`) reverted before submission.
- The predictions were formatted and saved as `submission.csv` for final evaluation.

### Key Takeaways
- Data preprocessing significantly improved model accuracy by addressing inconsistencies and missing values in `Item_Weight`, `Outlet_Size`, and `Item_Visibility`.
- Feature engineering helped extract meaningful patterns from `Outlet_Age`, `Price_per_Unit_Weight`, and `Item_Type_Grouped`, improving predictive performance.
- `CatBoost` outperformed other models, proving effective in handling categorical data (`Outlet_Size`, `Outlet_Type`) and complex interactions.
- Hyperparameter tuning to refine the final model for better accuracy.

### Conclusion
The final `CatBoost` model achieved an RMSE of **0.5206**, demonstrating strong predictive accuracy in forecasting sales. 

