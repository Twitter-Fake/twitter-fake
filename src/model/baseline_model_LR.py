import load_baseline_train_data
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.classification import accuracy_score,classification_report


df = load_baseline_train_data.get_baseline_data()

Y = df['label']
X = df.drop(columns= ['label'])


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42,stratify=Y)

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X_train,y_train)
print(X.columns.values)
print(LR.coef_[0])
predictions = LR.predict(X_test)
score = accuracy_score(y_test,predictions)
#score = LR.score(X_test, y_test)
print(score)
print(classification_report(y_test,predictions))

