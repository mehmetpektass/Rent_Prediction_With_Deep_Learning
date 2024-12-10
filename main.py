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
cols = ['img-link href', 'photo-count','list-view-date', 'list-view-title', 'left' , 'he-lazy-image src',  'img-wrp href', 'he-lazy-image src 2', 'listing-card--owner-info__firm-name', 'listing-card--owner-info__name', 'img-wrp href 2', 'he-lazy-image src 3', 'wp-btn']
drop_columns(df, cols) 
df.info() 


# Extract location parts (city, district, neighborhood) from 'list-view-location' column
df["city"] = df["list-view-location"].str.split("/").str[0]
df["district"] = df["list-view-location"].str.split("/").str[1]
df["neighborhood"] = df["list-view-location"].str.split("/").str[2]
df["neighborhood"] = df["neighborhood"].apply(lambda x: " ".join(x))

df.info()  
df.head() 


# Drop the 'list-view-location and city' column after splitting its values
drop_columns(df, ["list-view-location"])
drop_columns(df, ["city"])


# Clean and transform 'celly' column
df["celly"] = df["celly"].apply(lambda x: x.replace("Stüdyo" , "1+0"))
df["celly"] = df["celly"].apply(lambda x: x.replace("\n" , ""))
df["room"] = df["celly"].apply(lambda x: x.split("+")[0]).astype(int)
df["living_room"] = df["celly"].apply(lambda x: x.split("+")[1]).astype(int)

df["room"].unique()


# Clean and transform 'celly 2' column
df["celly 2"] = df["celly 2"].apply(lambda x: x.replace("." , ""))
df["area"] = df["celly 2"].apply(lambda x: x.split(" ")[0]).astype(int)

df["area"].unique()


# Clean and transform "celly 3" column
df["celly 3"].unique()
df["celly 3"] = df["celly 3"].apply(lambda x: str(x).replace("Sıfır Bina" , "0 Yaşında"))
df["celly 3"] = df["celly 3"].apply(lambda x: x.replace("\n" , " "))
df["age"] = df["celly 3"].apply(lambda x: x.split(" ")[0])

print(df["age"].unique())

df.info()
drop_columns(df, ["celly", "celly 2", "celly 3"])

