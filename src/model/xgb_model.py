from xgboost import XGBClassifier
from util import get_dataset
from sklearn.metrics import classification_report 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

def rf_grid_search(X,Y):
    param_grid = [
        {'n_estimators': [50,100,125], 'max_depth': [3,4,5]}
    ]
    #param_grid = [
        #   {'n_estimators': [750], 'max_depth': [2], 'min_samples_split': [2]},
    #]
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
    clf = GridSearchCV(XGBClassifier(),param_grid,cv=5,scoring=scoring,refit='AUC',n_jobs=3,verbose=1)
    clf.fit(X,Y)
    return clf
def train_xgb():

    train_df, test_df = get_dataset('lda')
    train_y = train_df.label
    test_y = test_df.label
    train_df.drop(['label', 'id', 'tweet', 'verified', 'Unnamed: 0', 'tweet'], axis = 1, inplace = True)
    test_df.drop(['label', 'id', 'tweet', 'verified', 'Unnamed: 0'], axis = 1, inplace = True)

    print(list(train_df))

    model = XGBClassifier()
    model.fit(train_df, train_y)
    predict = model.predict(test_df)
    for name, imp in zip(list(test_df), model.feature_importances_):
        print(name+":"+str(imp))
    #print(model.feature_importances_)
    ## print
    print(classification_report(test_y, predict))
    

if __name__ == "__main__":
    train_xgb()
    
