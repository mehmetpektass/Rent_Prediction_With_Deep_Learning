import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

df["district"] = df["district"].astype("category")
df["neighborhood"] = df["neighborhood"].astype("category")
df['room'] = df['room'].astype('int')
df['living_room'] = df['living_room'].astype('int')
df['area'] = df['area'].astype('int')
df['age'] = df['age'].astype('int')
df['floor'] = df['floor'].astype('int')
df['price'] = df['price'].astype('int')

df.describe()


columns = df.select_dtypes(include=[np.number]).columns
min_values = []
max_values = []
for column in columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    min_value = Q1 - 1.5*IQR
    max_value = Q3 + 1.5*IQR
    min_values.append(min_value)
    max_values.append(max_value)
    print(f"Column: {column}, min: {min_value}, max: {max_value}")