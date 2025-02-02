# BigMart Sales Prediction!

## Sales Prediction for Big Mart Outlets
The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and predict the sales of each product at a particular outlet.

Using this model, BigMart will try to understand the properties of products and outlets which play a key role in increasing sales.

Please note that the data may have missing values as some stores might not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Data Description](#data-description)
- [Evaluation Metric](#evaluation-metric)
- [Approach](#approach)
- [Source](#source)

## Problem Statement
Retail companies, such as BigMart, have extensive sales data. Analyzing this data can provide insights into customer purchasing behavior and inventory management. The objective of this competition is to predict the sales of products in various stores based on historical data. Participants must develop regression models to forecast `Item_Outlet_Sales` given a dataset of product and store features.

## Data Description
The dataset consists of historical sales data from different stores. It is divided into training and testing datasets:

### Training Dataset (`train.csv`)
This dataset contains 8523 rows and the following columns:

| Variable | Description |
|----------|-------------|
| `Item_Identifier` | Unique product ID |
| `Item_Weight` | Weight of product |
| `Item_Fat_Content` | Whether the product is low fat or not |
| `Item_Visibility` | The % of total display area of all products in a store allocated to the particular product |
| `Item_Type` | The category to which the product belongs |
| `Item_MRP` | Maximum Retail Price (list price) of the product |
| `Outlet_Identifier` | Unique store ID |
| `Outlet_Establishment_Year` | The year in which store was established |
| `Outlet_Size` | The size of the store in terms of ground area covered |
| `Outlet_Location_Type` | The type of city in which the store is located |
| `Outlet_Type` | Whether the outlet is just a grocery store or some sort of supermarket |
| `Item_Outlet_Sales` | Sales of the product in the particular store (Target variable to be predicted) |

### Test Dataset (`test.csv`)
Contains the same features as the training set, excluding the `Item_Outlet_Sales` column, which needs to be predicted.

## Submission File Format
The final submission file must contain the following variables:

| Variable | Description |
|----------|-------------|
| `Item_Identifier` | Unique product ID |
| `Outlet_Identifier` | Unique store ID |
| `Item_Outlet_Sales` | Predicted sales of the product in the particular store |

## Evaluation Metric
Your model performance will be evaluated on the basis of your prediction of the sales for the test data (`test.csv`), which contains similar data points as the training set except for the sales to be predicted. Your submission needs to be in the format shown in the sample submission.

We will use the **Root Mean Squared Error (RMSE)** to judge your response:

\[ RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \]

where:
- `y_i` is the actual sales value
- `\hat{y}_i` is the predicted sales value
- `n` is the number of observations

### Public and Private Split
The test file is further divided into:
- **Public (25%)** - Initial responses will be checked and scored on this portion.
- **Private (75%)** - Final rankings will be based on this score, which will be published once the competition is over.

## Approach
The following steps outline the approach to solving this problem:
1. **Exploratory Data Analysis (EDA)** - Understanding data distribution, missing values, and feature correlations.
2. **Data Preprocessing** - Handling missing values, encoding categorical variables, and feature scaling.
3. **Feature Engineering** - Creating new features based on domain knowledge.
4. **Model Selection** - Trying multiple regression models such as Linear Regression, Decision Trees, Random Forests, Gradient Boosting, and XGBoost.
5. **Hyperparameter Tuning** - Using techniques like GridSearchCV and RandomizedSearchCV.
6. **Model Evaluation** - Measuring performance using RMSE on the validation dataset.
7. **Submission** - Generating predictions for the test dataset and submitting the results.


## Source
This problem statement and dataset originate from [Analytics Vidhya](https://www.analyticsvidhya.com/datahack/contest/practice-problem-big-mart-sales-iii/). Please refer to their website for the official problem details and data access.
