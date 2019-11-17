"""
    This function get all the featueres to online processing.
"""
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import os

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

from bert_serving.client import BertClient
"""
    Common text processing functionalities.
"""
def general_text_processing(data):

    regex_list = [('\S*@\S*\s?', ''), 
        ('\s+', ' '),
        ("\'", "")
    ]

    for regex_text in regex_list:
        data = re.sub(regex_text[0], regex_text[1], data)
    return data
"""
    Remove stopwords
"""
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    valid_words = [x for x in re.findall(r"[\w]+", text) if x not in stop_words]
    return " ".join(valid_words)
    
"""
    Fill dictionary
"""
def fill_lda_result(df, lda_model, dictionary, topic_count):
    values = df['tweet'].values.tolist()
    doc2_corupus = [dictionary.doc2bow(text.split()) for 
        text in values]
    predicted_values = [lda_model[vec] for vec in doc2_corupus]
    """
        append to column
    """
    for i in range(len(predicted_values)):
        temp = [ 0 for x in range(topic_count)]
        for ele in predicted_values[i]:
            temp[ele[0]] = ele[1]
        predicted_values[i] = temp
    
    for index in range(topic_count):
        col_name = "topic_" + str(index)
        df[col_name] = [x[index] for x in predicted_values]
    
    return df
    



"""
    Topic modeling features.
"""
def topic_model(df_train, df_test, topic_count = 10):    
    ## general remove text
    df_train['tweet'] = df_train['tweet'].fillna("")
    df_test['tweet'] = df_test['tweet'].fillna("")

    df_train['tweet'] = df_train['tweet'].map(general_text_processing)
    df_test['tweet'] = df_test['tweet'].map(general_text_processing)

    ## remove stop words
    df_train['tweet'] = df_train['tweet'].map(remove_stop_words)
    df_test['tweet'] = df_test['tweet'].map(remove_stop_words)

    ## gensim lda
    dictionary = Dictionary()
    for t in df_train.tweet.values.tolist():
        #print(t)
        dictionary.add_documents([t.split()]) 
    #for  t in df_test['tweet'].values.tolist() :
        #print(t)
        # print(t[0].split())
        #print(dictionary.doc2bow(t.split()))
    train_doc2_corupus = [dictionary.doc2bow(text.split()) for 
        text in df_train['tweet'].values.tolist()]
    #print(train_doc2_corupus)
    print("Started LDA")
    lda_model = LdaModel(train_doc2_corupus, num_topics = topic_count, iterations = 30 )
    print("Completed LDA")
    

    """
    fill topics
    """
    df_test = fill_lda_result(df_test, lda_model, dictionary,
        topic_count )
    df_train = fill_lda_result(df_train, lda_model, dictionary,
        topic_count)
    

    """
    return 
    """
    print('LDA Completed')
    return df_train, df_test

"""
    Load the glove 2 vec
"""

def load_glov_vec(glove_file):

    mappings = {}
    with open(glove_file) as file:
        for line in file:
            splits = line.split()
            mappings[ splits[0] ] = splits[1:]
    return mappings

"""
Gensim average word encoding.
@input: dataframe with column tweet
@output: dataframe with averge word_count
"""

def glove_encode(df, glove_file, dims = 27):
    glove_model = load_glov_vec(glove_file)

    ## create representation
    tweets = df['tweet'].values.tolist()
    mappings = []
    

    """
        Get the tweet
    """
    for t in tweets:
        cur = [0 for x in range(dims)]
        size = 0
        for word in t.split():
            word = word.lower()
            if word in glove_model:
                temp_vec = glove_model[word]
                #print(temp_vec)
                for i in range(dims):
                    cur[i] += float(temp_vec[i])
                size += 1
        if size != 0:
            for i in range(dims):
                cur[i] /= size
        mappings.append(cur)
    """
        append dataframe
    """
    for i in range(dims):
        col_name = 'glove_' + str(i)
        df[col_name] = [ x[i] for x in mappings]


    return df
    
def bert_encode(df, dims=768, bert_csv_path='../data/bert.csv'):

    bc = BertClient()

    '''
        process null values
    '''
    df['tweet'] = df.tweet.fillna('_')
    tweets = df['tweet'].values.tolist()

    '''
        call service for bert embeddings
    '''
    bert_embeddings = bc.encode(tweets)

    for i in range(dims):
        col_name = 'bert_' + str(i)
        df[col_name] = [x[i] for x in bert_embeddings]
    
    '''
        save df
    '''
    df.to_csv(bert_csv_path)

    return df



if __name__ == "__main__":
    """
        Test lda
    """
    test_src = '../data/training_user_tweet.csv'
    df = pd.read_csv(test_src)
    df.tweet.fillna("", inplace = True)
    n_limt = int(df.shape[0]*0.8)
    df_train = df[0:n_limt]
    df_test = df[n_limt:]
    df_train, df_test = topic_model(df_train, df_test)
    print("######## test LDA  #####")
    print(list(df_train))
    print(list(df_test))



    # print("################ test glove ###### ")
    # glove_file = '/media/shibin/disk/glove/glove.twitter.27B/glove.twitter.27B.25d.txt'
    # glove_df = glove_encode(df_test, glove_file, 25 )
    # print(list(glove_df))
















