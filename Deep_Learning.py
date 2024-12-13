import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

df = pd.read_csv("data_cleaned.csv")


# Convert columns with loop of a function
categorical_features = ["district" , "neighborhood"]
numerical_features = ["room", "living_room", "area", "age", "floor"]

def convert_columns(df, columns,type):
    for column in columns:
        df[column] = df[column].astype(type)
        
convert_columns(df, categorical_features, "category")
convert_columns(df, numerical_features, int)


X = df.drop("price", axis=1)
y = df["price"]

# Create pipeline and fit-transform
full_pipeline = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

X_prepared = full_pipeline.fit_transform(X)

    

