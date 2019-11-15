# Laod train data to baseline model
import pandas as pd
import numpy as np

# Baseline model
# All fearures

# Convert object columns ( text ones ) to whether they exist or not
def convert_object_columns(df):
    for column in df:
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            df[column] = df[column].apply(lambda x: 0 if x == np.nan else 1)
    return df


def handle_nan_number_columns(df):
    return df.fillna(0)


def drop_unnecessary_columns(df,drop_columns_list):
    return df.drop(columns=drop_columns_list)


def get_baseline_data(filename= '../data/training_users.csv'):
    drop_columns = ['Unnamed: 0', 'lang', 'id', 'name', 'screen_name','protected','verified','updated','file']

    df = pd.read_csv(filename)

    df = convert_object_columns(df)
    df = handle_nan_number_columns(df)
    df = drop_unnecessary_columns(df, drop_columns)
    return df



def get_user_tweet_data(filename = '../data/training_user_tweet.csv'):
    df = pd.read_csv(filename)
    drop_columns = ['lang', 'id', 'name', 'screen_name', 'protected', 'verified', 'updated', 'file']
    # Remove tweet Null
    # Remove NaN tweets
    df = df.dropna(subset=['tweet'])
    tweetColumn = df['tweet'].values
    df.drop('tweet', axis=1,inplace=True)
    df = convert_object_columns(df)
    df = handle_nan_number_columns(df)
    df = drop_unnecessary_columns(df, drop_columns)
    df['tweet'] = tweetColumn
    return df





