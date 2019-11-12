
import pandas as pd
from sklearn.model_selection import train_test_split

data_no_additional_feature = '../data/training_users.csv'
data_with_tweet_csv = '../data/training_user_tweet.csv'
data_with_lsa = ''
from data import online_features
import numpy as np
"""
  Remove the field with objects
"""
def remap_fields(df):

    for name, dtype in zip(list(df), df.dtypes):
        
        if dtype == 'object':
            df[name] = df[name].map( lambda x: 1 if  x else 0)
    df.fillna(0, inplace = True)
    return df
            


"""
    get unified function get dataset
"""

def get_dataset(data_type='none'):

    if data_type == 'none':
        data = pd.read_csv(data_no_additional_feature)
        data = remap_fields(data)
        train_x, test_x, _, _ = train_test_split(data, data.label,  stratify =data.label)
        return train_x, test_x
    elif data_type == 'lda':
        data = pd.read_csv(data_with_tweet_csv)
        train_x, test_x, _, _ = train_test_split(data, data.label,  stratify =data.label)

        train_x , test_x= online_features.topic_model(train_x, test_x)
        train_x = remap_fields(train_x)
        test_x = remap_fields(test_x)
        return train_x, test_x
    



