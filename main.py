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
df["district"] = df["list-view-location"].str.split("/").str[1]
df["neighborhood"] = df["list-view-location"].str.split("/").str[2]
df["neighborhood"] = df["neighborhood"].str.split("Mah.").str[0].str.strip()



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


# Clean and transform "celly 3" column and drop some columns
df["celly 3"] = df["celly 3"].apply(lambda x: str(x).replace("Sıfır Bina" , "0 Yaşında"))
df["celly 3"] = df["celly 3"].apply(lambda x: x.replace("\n" , " "))
df["age"] = df["celly 3"].apply(lambda x: x.split(" ")[0])
df["age"] = df["age"].apply(lambda x: x.replace("nan", "0") if isinstance(x, str) else x)

print(df["age"].unique())
drop_columns(df, ["celly", "celly 2", "celly 3"])


# Clean and transform "celly 4" column and drop a column
replace_dict = {
    'Kot 2': '-2. Kat',
    'Kot 1': '-1. Kat',
    'Yüksek Giriş': '1. Kat',
    'Ara Kat': '3. Kat',
    'En Üst Kat': '5. Kat',
    'Bahçe Katı': '0. Kat',
    'Yarı Bodrum': '0. Kat',
    'Bodrum': '0. Kat',
    'Kot 3': '-3. Kat',
    'Çatı Katı': '5. Kat',
    'Zemin': '0. Kat',
    'Giriş Katı': '0. Kat',
    'Villa Katı': '0. Kat',
    '21 ve üzeri': '21. Kat',
    'Bodrum ve Zemin': '0. Kat',
    'Asma Kat': '1. Kat',
    'Tripleks': '0. Kat',
    'Teras Katı': '5. Kat',
    'nan': '2. Kat',
}
df["celly 4"] = df["celly 4"].replace(replace_dict.keys(), replace_dict.values()).astype(str)
df["floor"] = df["celly 4"].apply(lambda x: x.split(".")[0]).astype(int)
print(df['floor'].unique())

drop_columns(df, ["celly 4"])


# Clean and transform "list-view-price" column and drop a column
df['list-view-price'] = df['list-view-price'].astype(str).apply(lambda x: x.replace('.', ''))
df['price'] = df['list-view-price'].astype(int)
df = df[df["price"] != "NaN"]
print(df['price'].unique())

drop_columns(df, ["list-view-price"])


df.to_csv("data.csv" , index=False)