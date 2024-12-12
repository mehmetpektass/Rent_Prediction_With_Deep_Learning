import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv("data_cleaned.csv")


# This ensures consistent data types for numerical operations
df["district"] = df["district"].astype("category")
df["neighborhood"] = df["neighborhood"].astype("category")
df['room'] = df['room'].astype('int')
df['living_room'] = df['living_room'].astype('int')
df['area'] = df['area'].astype('int')
df['age'] = df['age'].astype('int')
df['floor'] = df['floor'].astype('int')
df['price'] = df['price'].astype('int')

df.info()


# Create pipeline and model
categorical_features = ["district", "neighborhood"]
numerical_features = ["room", "living_room", "area", "age", "floor"]

full_pipeline = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(), categorical_features)
])

model = Pipeline([
    ("preprocessing", full_pipeline),
    ("model", LinearRegression())
])


# Split test and train sets
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)