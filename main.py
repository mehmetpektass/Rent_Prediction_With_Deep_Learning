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
        except:
            print("An error happened while trying to drop the columns")


cols = ['img-link href', 'photo-count','list-view-date', 'list-view-title', 'left' , 'he-lazy-image src',  'img-wrp href', 'he-lazy-image src 2', 'listing-card--owner-info__firm-name', 'listing-card--owner-info__name', 'img-wrp href 2', 'he-lazy-image src 3', 'wp-btn']
drop_columns(df, cols)

print(df.info())
df.head(5)
