import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

# This ensures consistent data types for numerical operations
df["district"] = df["district"].astype("category")
df["neighborhood"] = df["neighborhood"].astype("category")
df['room'] = df['room'].astype('int')
df['living_room'] = df['living_room'].astype('int')
df['area'] = df['area'].astype('int')
df['age'] = df['age'].astype('int')
df['floor'] = df['floor'].astype('int')
df['price'] = df['price'].astype('int')

df.describe()
df.info()

# Calculate the interquartile range (IQR) for each numerical column
columns = df.select_dtypes(include=[np.number]).columns
min_values = []
max_values = []
for column in columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    min_value = Q1 - 1.5*IQR
    max_value = Q3 + 1.5*IQR
    if column == "price":
        max_value = 100000
    min_values.append(min_value)
    max_values.append(max_value)
    print(f"Column: {column}, min: {min_value}, max: {max_value}")
    

# Filter the DataFrame to remove outliers and filter price by a number
for i, column in enumerate(columns):
    df = df[(df[column] <= max_values[i]) & (df[column] >=min_values[i])]
    
df = df[df["price"] >= 2500]

df.to_csv('data_cleaned.csv', index=False)
