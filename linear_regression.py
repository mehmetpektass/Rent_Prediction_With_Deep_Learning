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
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = Pipeline([
    ("preparation", full_pipeline),
    ("model", LinearRegression())
])


# Split test and train sets
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Fit train sets and create scores
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_pred, y_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_pred, y_test)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")


#Check importance of features
feature_importances = model.named_steps['model'].coef_
print(len(feature_importances))
print(feature_importances)


# Loop through numerical features to display their coefficients
print("Numerical Features")
for i in range(len(numerical_features)):
    print(numerical_features[i], feature_importances[i])
    
    
# Loop through categorical features and display their coefficients    
print("Categorical Features")
for i in range(len(categorical_features)):
    for j in range(len(model.named_steps['preparation'].transformers_[1][1].categories_[i])):
        print(model.named_steps['preparation'].transformers_[1][1].categories_[i][j], feature_importances[len(numerical_features) + j])
        
        
new_data = pd.DataFrame({
    'district': ['Beşiktaş'],
    'neighborhood': ['Balmumcu Mah.'],
    'room': [4],
    'living_room': [1],
    'area': [130],
    'age': [15],
    'floor': [1]
}) 
print(model.predict(new_data))