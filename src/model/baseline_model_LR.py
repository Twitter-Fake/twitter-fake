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

# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)
#
# param = {
#     'max_depth': 4,  # the maximum depth of each tree
#     'eta': 0.3,  # the training step for each iteration
#     'silent': 0,  # logging mode - quiet
#     'objective': 'binary:logistic',  # error evaluation for multiclass training
#     }  # the number of classes that exist in this datset
# num_round = 20  # the number of training iterations
#
# bst = xgb.train(param, dtrain, num_round)
#
# preds = bst.predict(dtest)
# best_preds = np.asarray([np.argmax(line) for line in preds])
# print(accuracy_score(y_test,best_preds))

