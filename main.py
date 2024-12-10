import numpy as np
import pandas as pd


# Load CSV data into a DataFrame
df = pd.read_csv("scraped_data.csv")
df.info()  
df.head()  


# Function to drop specified columns from the DataFrame
def drop_columns(df, columns):
    for column in columns:
        try:
            df.drop([column], axis=1, inplace=True) 
        except:
            print("An error happened while trying to drop the columns")


# List of columns to drop
cols = ['img-link href', 'photo-count', 'list-view-date', ...]
drop_columns(df, cols) 
df.info() 


# Extract location parts (city, district, neighborhood) from 'list-view-location' column
df["city"] = df["list-view-location"].str.split("/").str[0]
df["district"] = df["list-view-location"].str.split("/").str[1]
df["neighborhood"] = df["list-view-location"].str.split("/").str[2]
df["neighborhood"] = df["neighborhood"].apply(lambda x: " ".join(x))

df.info()  
df.head() 


# Drop the 'list-view-location' column after splitting its values
drop_columns(df, ["list-view-location"])
drop_columns(df, ["city"])

