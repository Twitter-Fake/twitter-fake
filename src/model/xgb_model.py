from xgboost import XGBClassifier
from util import get_dataset
from sklearn.metrics import classification_report 
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
    