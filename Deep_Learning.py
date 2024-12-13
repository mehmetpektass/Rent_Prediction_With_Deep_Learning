import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
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


#Split the dataset as X nad y
X = df.drop("price", axis=1)
y = df["price"]


# Create pipeline and fit-transform and finally train-test sets
full_pipeline = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

X_prepared = full_pipeline.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42)

    
# Define the neural network model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="linear"),
])


# Compile the model and train the model
model.compile(optimizer="adam",
              loss="mean_squared_error",
              metrics=["mae"])

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Mean Absolute Error: {mae}")

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")