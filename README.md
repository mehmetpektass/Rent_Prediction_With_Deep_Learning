# Real Estate Price Prediction & Data Processing Pipeline🏠

## Description
This project is a comprehensive real estate data processing and prediction pipeline. It includes web-scraped data cleaning, feature engineering, and machine learning models to predict house prices and classify properties into price ranges.

<br>

## Features

* **Web-Scraped Data Cleaning & Transformation**  
    * Removes unnecessary columns.
    * Extracts and standardizes categorical and numerical        features.
    * Converts Turkish floor naming conventions into numeric values.
    * Cleans and formats price data.
*  **Machine Learning Models** 
    * ***Neural Network (Deep Learning)***
        * A deep learning model using TensorFlow/Keras to predict house prices.
        * Uses a Sequential model with dense and dropout layers.
        * Trains on standardized numerical and one-hot-encoded categorical data.
        * Evaluates model performance using R² score and MAE.
    * ***Linear Regression Model***
    
        * Uses sklearn.pipeline to preprocess data and train a linear regression model.
        * Evaluates feature importance to understand price determinants.
        * Computes RMSE and R² score for performance evaluation.
    * ***Random Forest Classifier***
        * Categorizes house prices into different price ranges.
        * Uses a RandomForestClassifier to classify properties into price bins.
        * Evaluates classification performance using confusion matrix and precision-recall scores.
        
* **Data Preprocessing** 
    * Converts categorical data (district, neighborhood) into numerical representations via one-hot encoding.
    * Standardizes numerical features (area, age, floor, etc.).
    * Bins price values into predefined categories for classification.
* **Feature Importance Analysis** 
    * Extracts and analyzes feature importance from the trained models.
    * Identifies which features contribute most to price determination.
* **New Data Prediction**  
    * Accepts new property details as input.
    * Predicts the estimated price or price category using trained models.

<br>

## Tech Stack
### Tools and Libraries
* **Python**
* **Pandas**
* **NumPy**
* **Scikit-Learn:** 
    * train_test_split
    * StandardScaler
    * OneHotEncoder
    * ColumnTransformer
    * Pipeline
    * LinearRegression
    * RandomForestClassifier
    * classification_report, confusion_matrix
    * mean_squared_error, r2_score
* **TensorFlow/Keras**
    * Sequential
    * Dense
    * Dropout
<br>

<br>




## Installation & Usage



```bash
git clone https://github.com/mehmetpektass/Rent_Prediction_With_Deep_Learning.git

```
```
cd real-estate-analysis

```
<br>

#### 1. Data Cleaning
Run the following script to preprocess the raw data:
```
python3 data_processing.py
```
#### 2. Price Prediction (Neural Network)
```
python3 price_prediction_nn.py
```
#### 3. Regression Model
```
python3 regression_model.py
```
#### 4. Price Classification
```
python3 classification_model.py
```
<br>

## Contribution Guidelines  🚀
 Pull requests are welcome. If you'd like to contribute, please:

* Fork the repository.
* Create a feature branch.
* Submit a pull request with a clear description of changes.



