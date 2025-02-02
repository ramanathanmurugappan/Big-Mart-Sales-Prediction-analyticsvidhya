# Big Mart Sales Prediction

This project aims to predict sales for products across different Big Mart outlets using machine learning techniques.

## Project Structure

```
├── EDA.ipynb              # Exploratory Data Analysis notebook
├── preprocessing_utils.py   # Utility functions for data preprocessing
├── run_preprocessing.py     # Main preprocessing script
├── modelling.py            # Machine learning model implementation
├── modelling.ipynb        # Model development notebook
├── train_v9rqX0R.csv      # Raw training data
├── test_AbJTz2l.csv       # Raw test data
├── processed_train.csv    # Preprocessed training data
├── processed_test.csv     # Preprocessed test data
└── submission.csv         # Final predictions
```

## Project Files

1. **Data Files**
   - `train_v9rqX0R.csv`: Raw training dataset
   - `test_AbJTz2l.csv`: Raw test dataset
   - `sample_submission_8RXa3c6.csv`: Sample submission file format

2. **Notebooks and Analysis**
   - `EDA.ipynb`: Exploratory Data Analysis notebook containing:
     - Data profiling and visualization
     - Missing value analysis
     - Feature distributions and relationships
     - Outlet and item analysis
   - `modelling.ipynb`: Model development notebook with:
     - Model training experiments
     - Hyperparameter tuning
     - Feature importance analysis
     - Model evaluation

3. **Python Scripts**
   - `preprocessing_utils.py`: Contains utility functions for:
     - Data cleaning
     - Feature engineering
     - Missing value imputation
     - Feature scaling and encoding
   - `run_preprocessing.py`: Main script for data preprocessing pipeline
   - `modelling.py`: Script for training the final CatBoost model and generating predictions

4. **Results and Documentation**
   - `optuna_results.csv`: Hyperparameter optimization results
   - `random_search_results.csv`: Random search tuning results
   - `Hackathon.md`: Competition details and guidelines

