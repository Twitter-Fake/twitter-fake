import load_baseline_train_data
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.classification import accuracy_score,classification_report


# Grid Search for Twitter SVM
# def svc_param_selection(X, y, nfolds):
#     Cs = [0.001,0.1]
#     gammas = [0.001, 1]
#     kernels = ['linear','rbf']
#     param_grid = {'C': Cs, 'gamma' : gammas, 'kernel':kernels}
#     grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfolds)
#     grid_search.fit(X, y)
#     grid_search.best_params_
#     return grid_search.best_params_


df = load_baseline_train_data.get_baseline_data()

Y = df['label']
X = df.drop(columns= ['label'])


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42,stratify=Y)
print(svc_param_selection(X_train,y_train,4))
# model = svm(random_state=0, solver='lbfgs', multi_class='ovr').fit(X_train,y_train)
# print(X.columns.values)
# print(LR.coef_[0])
# predictions = LR.predict(X_test)
# score = accuracy_score(y_test,predictions)
# #score = LR.score(X_test, y_test)
# print(score)
# print(classification_report(y_test,predictions))