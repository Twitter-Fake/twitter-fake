
import pandas as pd
from sklearn.model_selection import train_test_split

data_no_additional_feature = '../data/training_users.csv'
data_with_tweet_csv = '../data/training_user_tweet.csv'
data_with_lsa = ''
import online_features_model
import numpy as np
import os
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

        train_x , test_x= online_features.lda_parallel(train_x, test_x, topic_count = 20 )
        train_x = remap_fields(train_x)
        test_x = remap_fields(test_x)
        return train_x, test_x
    elif data_type == 'bert':
        data = None
        if not os.path.exists('../data/bert.csv'):
            data = online_features_model.bert_encode(pd.read_csv(data_with_tweet_csv))
        else:
            data = pd.read_csv('../data/bert.csv')

        print(data.sample(2)) # with bert
        
        '''
            create test and train split
        '''
        data = remap_fields(data)
        train_x, test_x, _, _ = train_test_split(data, data.label,  stratify =data.label)
        train_y = train_x.label
        test_y = test_x.label

        train_x.drop(['Unnamed: 0', 'label', 'id', 'tweet', 'verified'], axis = 1, inplace = True)
        test_x.drop(['Unnamed: 0', 'label', 'id', 'tweet', 'verified'], axis = 1, inplace = True)

        return train_x, train_y, test_x, test_y



