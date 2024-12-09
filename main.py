import numpy as np
import pandas as pd

df = pd.read_csv("scraped_data.csv")
df.info()
df.head()


def drop_columns(df, columns):
    for column in columns:
        try:
            df.drop([column] , axis=1 , inplace=True)
            help(df.drop)
        except Exception as e:
            print("An error happened while trying to drop the columns")

