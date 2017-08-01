# coding=utf-8
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import accuracy_score

from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
#import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the Census dataset
data = pd.read_csv("census.csv")


# print data.head(n=10)
# print "--------------"
# print data.size
# print data.shape

print "Some tests...\n\n"
#
income_data = data['income']
print income_data.head(n=10)
#
# n_records = income_data.count()
# print n_records
#
# print list(data.columns)
# num_cols = data._get_numeric_data().columns
# print "---------------"
# print num_cols
# print "-----------------\n"
# print data[data.income==">50K"].income.count()

ct,cw =0,0
for m in data['income']:
    if m == ">50K":
        ct +=1
    else:
        cw +=1

print ct,cw
print data[data.income==">50K"].income.count()
print data[data.income=="<=50K"].income.count()

print data.income.count()

print "End testing ...................\n"

print "..............Log-transform the skewed data ........\n"

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))


print "..............Scaling log-transformed data ........\n"

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
print features_log_minmax_transform.head(n = 5)

print "..............One hot encoding ........\n"

# TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_final =  pd.get_dummies(features_log_minmax_transform)

# TODO: Encode the 'income_raw' data to numerical values
def nEnc(x):
    if x =='>50K': return 1
    else: return 0
income = income_raw.apply(nEnc)

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# Uncomment the following line to see the encoded feature names
#print encoded

"""Now all categorical variables have been converted into numerical features, and all numerical features have been 
normalized. As always, we will now split the data (both features and their labels) into training and test sets. 
80% of the data will be used for training and 20% for testing."""


# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final,income,test_size = 0.2,random_state = 0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


#-------------------------------------------------------- naive predictor ----------------------------------------------

'''
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case

'''

if False:

    print "----------------- NAIVE PREDICTOR BUILT BY HAND ------------------"


    TP = float(np.sum(income))
    FP = float(income.count())
    TN=0.0
    FN=0.0
    beta = 0.5

    print "TP={}, FP={}, TN={}, FN={} ".format(TP,FP,TN,FN)
    print "%%%%%%%%%%%%%"
    print "TP/FP = {}". format(float(TP/FP))
    print "%%%%%%%%%%%%%"

    # TODO: Calculate accuracy, precision and recall
    accuracy = (TP + TN)/(FP+FN+TP+TN)
    recall = TP/(TP+FN)
    precision = float(TP/(TP+FP))

    print "accuracy={}, recall={}, precision={} ".format(accuracy, recall, precision)

    # TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
    # HINT: The formula above can be written as (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    fscore = (1.0 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    # Print the results
    print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)
    print ""

    print "-------------------------------------NAIVE BAYES ---------------------------------------"

if False:

    #from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import GaussianNB
    #from sklearn import cross_validation


    from sklearn.metrics import confusion_matrix


    from sklearn.metrics import recall_score as recall
    from sklearn.metrics import precision_score as precision

    clfNB = GaussianNB()
    clfNB.fit(X_train,y_train)
    print "GaussianNB has accuracy: ",accuracy_score(y_test, clfNB.predict(X_test))
    cmatrixNB = confusion_matrix(y_test,clfNB.predict(X_test))
    print "GaussianNB confusion matrix:\n",cmatrixNB

    #for computing the f1 score
    from sklearn.metrics import f1_score

    NBf1 =f1_score(y_test, clfNB.predict(X_test))
    print "GaussianNB F1 score: {:.2f}".format(NBf1)



# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score (DONE!)

from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression




def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time()  # Get start time
    learner = learner.fit(X_train[:],y_train[:])
    end = time()  # Get end time

    # TODO: Calculate the training time
    results['train_time'] = end-start

    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time()  # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:sample_size])
    end = time()  # Get end time

    # TODO: Calculate the total prediction time
    results['pred_time'] = end-start

    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:sample_size], predictions_train)

    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] =fbeta_score(y_train[:sample_size], predictions_train, beta=0.5)

    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)

    # Success
    #print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)

    #Modification

    results['clf_name'] = learner.__class__.__name__
    results['samp_size'] = sample_size

    # Return the results
    return results


    # train_predict return value format: it returns a dictionary
    # {'samp_size': 36177, 'clf_name': 'GaussianNB', 'pred_time': 0.037,
    # 'f_test': 0.42, 'train_time': 0.038, 'acc_train': 0.59,
    # 'acc_test': 0.59, 'f_train': 0.42}


"""
Gaussian Naive Bayes (GaussianNB)
Decision Trees
Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
K-Nearest Neighbors (KNeighbors)
Stochastic Gradient Descent Classifier (SGDC)
Support Vector Machines (SVM)
Logistic Regression
"""


if False:

    gnb = GaussianNB()
    dt = DecisionTreeClassifier(random_state=0)
    bags = BaggingClassifier(KNeighborsClassifier(), random_state=10, n_jobs=6)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(random_state=20), algorithm="SAMME")
    rfc = RandomForestClassifier(random_state=90, n_jobs=6)
    gdb = GradientBoostingClassifier(random_state=30)
    knn = KNeighborsClassifier(n_jobs=6)
    stgd = SGDClassifier(random_state=40, n_jobs=6)
    lsvc = LinearSVC(random_state=80)
    svc =  SVC(random_state=60)
    lreg = LogisticRegression(random_state=50)

    print train_predict(gnb, len(y_train), X_train, y_train, X_test, y_test)
    print train_predict(dt, len(y_train), X_train, y_train, X_test, y_test)
    print train_predict(bags, len(y_train), X_train, y_train, X_test, y_test)
    print train_predict(bdt, len(y_train), X_train, y_train, X_test, y_test)
    print train_predict(rfc, len(y_train), X_train, y_train, X_test, y_test)
    print train_predict(gdb, len(y_train), X_train, y_train, X_test, y_test)
    print train_predict(knn, len(y_train), X_train, y_train, X_test, y_test)
    print train_predict(stgd, len(y_train), X_train, y_train, X_test, y_test)
    print train_predict(lsvc, len(y_train), X_train, y_train, X_test, y_test)
    print train_predict(svc, len(y_train), X_train, y_train, X_test, y_test)
    print train_predict(lreg, len(y_train), X_train, y_train, X_test, y_test)



if True:

    gnb = GaussianNB()
    dt = DecisionTreeClassifier(random_state=0)
    bags = BaggingClassifier(KNeighborsClassifier(), random_state=10, n_jobs=6)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(random_state=20), algorithm="SAMME")
    rfc = RandomForestClassifier(random_state=90, n_jobs=6)
    gdb = GradientBoostingClassifier(random_state=30)
    knn = KNeighborsClassifier(n_jobs=6)
    stgd = SGDClassifier(random_state=40, n_jobs=6)
    lsvc = LinearSVC(random_state=80)
    svc =  SVC(random_state=60)
    lreg = LogisticRegression(random_state=50)

    clf_list = [gnb, dt, bags,bdt, rfc,gdb,knn,stgd,lsvc, svc,lreg]

    clfl_short = [gnb,lsvc]

    clf_lscores =[]
    clf_lnames = []

    for i in range(len(clfl_short)):
        clf_results = train_predict(clfl_short[i], len(y_train), X_train, y_train, X_test, y_test)
        clf_lscores.append(clf_results['f_test'])
        clf_lnames.append(clf_results['clf_name'])



    n_groups = len(clf_lscores)
    impdata = clf_lscores
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4
    rects1 = plt.bar(index, impdata, bar_width, alpha=opacity, color='b')

    plt.xlabel('Classifier')
    plt.ylabel('F-Score Test ')
    plt.title('Classifier F-Score on Test Data Set')
    plt.xticks(index, clf_lnames)
    plt.legend()
    plt.tight_layout()
    plt.show()










if False:
    n_groups = len(reg.feature_importances_)
    impdata = reg.feature_importances_


    fig, ax = plt.subplots()

    index = np.arange(n_groups)

    bar_width = 0.35
    opacity = 0.4
    rects1 = plt.bar(index, impdata, bar_width, alpha=opacity, color='b', label='Feature Importance')

    plt.xlabel('Features')
    plt.ylabel('Gini Importance')
    plt.title('Feature Importance for Home Price Forecasting in Boston in 1978')
    plt.xticks(index, list(X_train.columns))
    plt.legend()
    plt.tight_layout()
    plt.show()






if False:

    print ""
    print ""

    #bdt = AdaBoostClassifier(LinearSVC(random_state=10),
    #                         algorithm="SAMME",
    #                         n_estimators=200)
    #print train_predict(bdt, 300, X_train, y_train, X_test, y_test)

    #bdt2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8, random_state=20),
    #                          algorithm="SAMME",
    #                          n_estimators=200)

    #print train_predict(bdt2, 300, X_train, y_train, X_test, y_test)


    grbs1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=10,random_state=30,max_depth=8)
    grbs2 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=20, random_state=30, max_depth=8)
    grbs3 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=20, random_state=30, max_depth=4)

    print train_predict(grbs1, len(y_train), X_train, y_train, X_test, y_test)
    print train_predict(grbs2, len(y_train), X_train, y_train, X_test, y_test)
    print train_predict(grbs3, len(y_train), X_train, y_train, X_test, y_test)






if False:




    stgd = SGDClassifier(alpha=0.01, loss="hinge", learning_rate='optimal', random_state=40, shuffle=True, n_jobs=4)

    print train_predict(stgd, 300, X_train, y_train, X_test, y_test)

    dt = DecisionTreeClassifier(max_depth=8,random_state=0)

    print train_predict(dt, 300, X_train, y_train, X_test, y_test)

    rfc = RandomForestClassifier(random_state=90, max_depth=16, n_jobs=4)

    print train_predict(rfc, 300, X_train, y_train, X_test, y_test)

    lin_svc =LinearSVC(random_state=30)

    print train_predict(lin_svc, 300, X_train, y_train, X_test, y_test)





    #print train_predict(linear_model.LogisticRegression(C=1e5),300, X_train, y_train, X_test, y_test)

    #bags = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5, random_state=10, n_jobs=4)

    #print train_predict(bags, 300, X_train, y_train, X_test, y_test)
    #print train_predict(GaussianNB(), 300, X_train, y_train, X_test, y_test)
    #





"""Implementation: Initial Model Evaluation
In the code cell, you will need to implement the following:
Import the three supervised learning models you've discussed in the previous section.
Initialize the three models and store them in 'clf_A', 'clf_B', and 'clf_C'.
Use a 'random_state' for each model you use, if provided.
Note: Use the default settings for each model — you will tune one specific model in a later section.
Calculate the number of records equal to 1%, 10%, and 100% of the training data.
Store those values in 'samples_1', 'samples_10', and 'samples_100' respectively.
Note: Depending on which algorithms you chose, the following implementation may take some time to run!
"""



if False:

    print ""
    print "------------------------------ Pipeline Implementation ---------------------------------\n"
    print ""
    print "Length of y_train = {}".format(len(y_train))

    print "------------*****----------Pipeline Test ----------*******-----------------------"

    # TODO: Import the three supervised learning models from sklearn (!DONE Above)

    # TODO: Initialize the three models
    clf_A = LinearSVC(random_state=10)
    clf_B = DecisionTreeClassifier(max_depth=8,random_state=0)
    clf_C = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4,random_state=20),
                         algorithm="SAMME",
                         n_estimators=200)

    #Adding more

    clf_D = GradientBoostingClassifier(n_estimators=10,random_state=30,max_depth=8)
    #clf_D = RandomForestClassifier(random_state=90, max_depth=16, n_jobs=4)
    #clf_E = AdaBoostClassifier(RandomForestClassifier(random_state=90, max_depth=16, n_jobs=4),
                         #algorithm="SAMME",
                         #n_estimators=200)

    # TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
    # HINT: samples_100 is the entire training set i.e. len(y_train)
    # HINT: samples_10 is 10% of samples_100
    # HINT: samples_1 is 1% of samples_100

    samples_100 = int(len(y_train))
    samples_10 = int(0.1*samples_100)
    samples_1 = int(0.01*samples_100)

    # Collect results on the learners
    results = {}
    for clf in [clf_A, clf_B, clf_C, clf_D]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([samples_1, samples_10, samples_100]):
            results[clf_name][i] = \
            train_predict(clf, samples, X_train, y_train, X_test, y_test)

    # Run metrics visualization for the three supervised learning models chosen
    print "-------- results, accuracy, fscore:"
    print results, accuracy, fscore


#--------------------------- GRID SEARCH OPTIMIZATION ---------------------------------

if False:
    print ""
    print "**************************** GRID SEARCH ***************************"
    print ""

    # TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer

    from sklearn.metrics import fbeta_score


    def performance_metric(y_true, y_predict):
        """ Calculates and returns the performance score between
            true and predicted values based on the metric chosen. """

        # TODO: Calculate the performance score between 'y_true' and 'y_predict'
        score = fbeta_score(y_true, y_predict,beta=0.5)

        # Return the score
        return score



    # TODO: Initialize the classifier
    #clf = GradientBoostingClassifier(n_estimators=10,random_state=30,max_depth=8)
    clf = GradientBoostingClassifier(random_state=30)

    # TODO: Create the parameters list you wish to tune, using a dictionary if needed.
    # HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}

    #parameters = {'max_depth':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'n_estimators':[5,10,50,100,200],'min_samples_split':[2,3,4,8,10]}

    parameters = {'max_depth': [4,6, 8, 10], 'n_estimators': [10, 100, 200]}

    # TODO: Make an fbeta_score scoring object using make_scorer()
    scorer = make_scorer(fbeta_score,beta=0.5)
    #scoring_fnc = make_scorer(performance_metric)

    # TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
    grid_obj = GridSearchCV(clf, parameters,scoring=scorer)

    # TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
    grid_fit = grid_obj.fit(X_train,y_train)

    # Get the estimator
    best_clf = grid_fit.best_estimator_

    # Make predictions using the unoptimized and model
    predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

    # Report the before-and-afterscores
    print "Unoptimized model\n------"
    print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
    print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5))
    print "\nOptimized Model\n------"
    print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
    print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))


##*********************** FEATURE IMPORTANCE ****************

if False:

    # TODO: Import a supervised learning model that has 'feature_importances_'


    # TODO: Train the supervised model on the training set using .fit(X_train, y_train)
    model = GradientBoostingClassifier(n_estimators=10,random_state=30,max_depth=8)
    model.fit(X_train,y_train)

    # TODO: Extract the feature importances using .feature_importances_
    importances = model.feature_importances_

    print importances

#----------------------------- visuals ----------------------



def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.

    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """

    # Create figure
    fig, ax = pl.subplots(2, 3, figsize=(11, 7))

    # Constants
    bar_width = 0.3
    colors = ['#A00000', '#00A0A0', '#00A000']

    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                # Creative plot code
                ax[j / 3, j % 3].bar(i + k * bar_width, results[learner][i][metric], width=bar_width, color=colors[k])
                ax[j / 3, j % 3].set_xticks([0.45, 1.45, 2.45])
                ax[j / 3, j % 3].set_xticklabels(["1%", "10%", "100%"])
                ax[j / 3, j % 3].set_xlabel("Training Set Size")
                ax[j / 3, j % 3].set_xlim((-0.1, 3.0))

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[0, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=learner))
    pl.legend(handles=patches, bbox_to_anchor=(-.80, 2.53), \
              loc='upper center', borderaxespad=0., ncol=3, fontsize='x-large')

    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize=16, y=1.10)
    pl.tight_layout()
    pl.show()