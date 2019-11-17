import load_baseline_train_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.classification import accuracy_score,classification_report


def clean_tweet(x):
    #Create a string form of our list of text
    raw_string = ''.join(x)
    no_links = re.sub(r'http\S+', '', raw_string)
    no_unicode = re.sub(r"\\[a-z][a-z]?[0-9]+", '', no_links)
    no_special_characters = re.sub('[^A-Za-z ]+', '', no_unicode)
    words = no_special_characters.split(" ")
    words = [w for w in words if len(w) > 2]
    words = [w.lower() for w in words]
    stw = stopwords.words('english')
    words = [w for w in words if w not in stw]
    print('done')
    return words


df = load_baseline_train_data.get_user_tweet_data()

Y = df['label']
X = df.drop(columns= ['label'])
df['tweet'] = df['tweet'].apply(clean_tweet)

# Clustering of Genuine Tweet
tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 2000,
    stop_words = 'english'
)
tfidf.fit(df.tweet)
text = tfidf.transform(df.tweet)
print('TF-IDF fitting complete')
print(text.shape)
print(df.shape)


X = df.drop(columns= ['tweet'])
Y = X['label']
X = X.drop(columns= ['label'])
X = pd.concat([X, pd.DataFrame(text.todense())], axis=1,ignore_index=True)
print(X.shape)
print(Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42,stratify=Y)
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)


# LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X_train,y_train)
# print(X.columns.values)
# print(LR.coef_[0])
predictions = svclassifier.predict(X_test)
score = accuracy_score(y_test,predictions)
# #score = LR.score(X_test, y_test)
print(score)
print(classification_report(y_test,predictions))



