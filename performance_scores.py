from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score

# returns proba predictions from logistics reg
def logistics_reg(X, y, Xtest):
    clf = LogisticRegression().fit(X, y)
    # print(clf.score(Xtest, Ytest))
    return clf.predict_proba(Xtest)[:,1]

def logistics_MAE_score(X, Y, Xtest, Ytest):
    Ytrain = Y > 0.5
    Ypred = logistics_reg(X, Ytrain, Xtest)
    # print(Ypred)
    # print(Ytest)
    return mean_absolute_error(Ytest, Ypred)

def logistics_accuracy_score(X, Y, Xtest, Ytest):
    Ytrain = Y > 0.5
    Ytrue = Ytest > 0.5
    Ypred = logistics_reg(X, Ytrain, Xtest)
    Yclass = Ypred > 0.5
    return accuracy_score(Ytrue, Yclass)

