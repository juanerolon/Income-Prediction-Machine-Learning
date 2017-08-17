
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from trainPredict import train_predict

def nEnc(x):
    if x =='>50K': return 1
    else: return 0

data = pd.read_csv("census.csv")
income_raw = data['income']

income_raw = data['income']


#----------------------------------- Full set of features -----------------------------------------------

features_raw = data.drop('income', axis = 1)
print "Full features results"
print "Number of raw features: {}".format(len(list(features_raw.columns)))

skewed = ['capital-gain', 'capital-loss']
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_log_transformed = pd.DataFrame(data=features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
scaler = MinMaxScaler()
features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
features_final = pd.get_dummies(features_log_minmax_transform)
income = income_raw.apply(nEnc)
X_train, X_test, y_train, y_test = train_test_split(features_final,income,test_size = 0.2,random_state = 0)

#select classifier to train of full feature set

#clf = GradientBoostingClassifier(n_estimators=10, random_state=30, max_depth=8)
#results =  train_predict(clf, len(y_train), X_train, y_train, X_test, y_test)

clf = GaussianNB()
results = train_predict(clf, len(y_train), X_train, y_train, X_test, y_test)

acc_trn0 = results['acc_train']
acc_tst0 = results['acc_test']
fsc_trn0 = results['f_train']
fsc_tst0 = results['f_test']

print "Training accuracy: {}".format(results['acc_train'])
print "Testing  accuracy: {}".format(results['acc_test'])
print "Training f-score : {}".format(results['f_train'])
print "Testing  f-score : {}".format(results['f_test'])


#----------------------------------- Feature selection test -----------------------------------------------

features = data.drop('income', axis = 1)
feat_names = list(features.columns)

for feat in feat_names:
    print ""
    print "Removing feature {}".format(feat)
    print ""
    features_raw = features.drop(feat, axis=1)
    #print "Number of raw features: {}".format(len(list(features_raw.columns)))

    skewed = ['capital-gain', 'capital-loss']
    if feat in skewed:
        skewed.remove(feat)

    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    if feat in numerical:
        numerical.remove(feat)

    features_log_transformed = pd.DataFrame(data=features_raw)
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
    scaler = MinMaxScaler()

    features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
    features_final = pd.get_dummies(features_log_minmax_transform)
    income = income_raw.apply(nEnc)
    X_train, X_test, y_train, y_test = train_test_split(features_final,income,test_size = 0.2,random_state = 0)



    #Naive Bayes
    clf = GaussianNB()
    results = train_predict(clf, len(y_train), X_train, y_train, X_test, y_test)

    #Gradient Boosting
    """
    clf = GradientBoostingClassifier(n_estimators=10, random_state=30, max_depth=8)
    results =  train_predict(clf, len(y_train), X_train, y_train, X_test, y_test)
    """

    acc_trn = results['acc_train']
    acc_tst = results['acc_test']
    fsc_trn = results['f_train']
    fsc_tst = results['f_test']

    #print "Training accuracy: {}   Change: {}".format(acc_trn, abs(acc_trn0-acc_trn))
    #print "Testing  accuracy: {}   Change: {}".format(acc_tst, abs(acc_tst0-acc_tst))
    #print "Training f-score : {}   Change: {}".format(fsc_trn, abs(fsc_trn0-fsc_trn))
    print "Testing  f-score : {}   Change: {}".format(fsc_tst, abs(fsc_tst0-fsc_tst))


    # Optional Grid search after classifier selection
    if False:

        clf = GradientBoostingClassifier(n_estimators=10,random_state=30,max_depth=8)
        #parameters = {'max_depth': [4,6, 8, 10], 'n_estimators': [200,300]}
        parameters = {}

        #clf = LogisticRegression(random_state=30, n_jobs=6)
        #parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}


        scorer = make_scorer(fbeta_score,beta=0.5)
        grid_obj = GridSearchCV(clf, parameters,scoring=scorer)
        grid_fit = grid_obj.fit(X_train,y_train)
        best_clf = grid_fit.best_estimator_
        predictions = (clf.fit(X_train, y_train)).predict(X_test)
        best_predictions = best_clf.predict(X_test)
        best_parameters = grid_fit.best_params_

        #print "Best grid search parameters: {}".format(best_parameters)
        print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
        print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))