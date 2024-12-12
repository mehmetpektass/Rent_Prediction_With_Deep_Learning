import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('data_cleaned.csv')

# This ensures consistent data types for numerical operations
df["district"] = df["district"].astype("category")
df["neighborhood"] = df["neighborhood"].astype("category")
df['room'] = df['room'].astype('int')
df['living_room'] = df['living_room'].astype('int')
df['area'] = df['area'].astype('int')
df['age'] = df['age'].astype('int')
df['floor'] = df['floor'].astype('int')
df['price'] = df['price'].astype('int')


# Create pipeline and model
categorical_features = ['city', 'district', 'neighborhood']
numerical_features = ['room', 'living_room', 'area', 'age', 'floor']


full_pipeline = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = Pipeline([
    ("preparation", full_pipeline),
    ("model", RandomForestClassifier(n_estimators=100))
])


# Split test and train sets
X = df.drop("price", axis=1)
y = df["price"]

bins = [x for x in range(0, 300000, 10000)]
labels = [x for x in range(1, 30)]
print(bins, labels)


