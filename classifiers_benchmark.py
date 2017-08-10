
#-----------------------------------------------------------------------
#
#Presents in a bar plot the benchmarks of some scikit learn classfiers.
#The classifiers are intialized using default parameters and hyperparameters
#It uses f-score and training time as scoring benchmarks
#It uses a census dataset to find donors for charity - Machine Learning Nanodegree
#
#Erique (Juan E) Rolon (https://github.com/juanerolon)
#
#-----------------------------------------------------------------------

import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt

#Local module import of train and predict utility

from trainPredict import train_predict

# Import the sklearn classifiers to be benchmarked

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


# Load the Census dataset
data = pd.read_csv("census.csv")

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

features_final = pd.get_dummies(features_log_minmax_transform)

# Encode the 'income_raw' data to numerical values
def nEnc(x):
    if x =='>50K': return 1
    else: return 0
income = income_raw.apply(nEnc)

# Set the number of features after one-hot encoding
encoded = list(features_final.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final,income,test_size = 0.2,random_state = 0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

#Instantiate classifier objects and stored them in a list for later access

gnb = GaussianNB()
dt = DecisionTreeClassifier(random_state=0)
bags = BaggingClassifier(KNeighborsClassifier(), random_state=10, n_jobs=6)
bdt = AdaBoostClassifier(DecisionTreeClassifier(random_state=20), algorithm="SAMME")
rfc = RandomForestClassifier(random_state=90, n_jobs=6)
gdb = GradientBoostingClassifier(random_state=30)
knn = KNeighborsClassifier(n_jobs=6)
stgd = SGDClassifier(random_state=40, n_jobs=6)
lsvc = LinearSVC(random_state=80)
svc = SVC(random_state=60)
lreg = LogisticRegression(random_state=50)

clf_list = [[gnb, dt, rfc, lsvc], [knn, stgd, lreg, bags], [bdt, gdb, svc]]

#Create lists to store selected benchmarking results
clf_lscores = []
clf_ltrain_times = []
clf_lnames = []

#Train all classifiers on training data and make predictions on test data
#Store desired benchmarking results
for j in range(len(clf_list)):
    temp_scores = []
    temp_times = []
    temp_names = []
    for i in range(len(clf_list[j])):
        clf_results = train_predict(clf_list[j][i], len(y_train), X_train, y_train, X_test, y_test)
        temp_scores.append(clf_results['f_test'])
        temp_times.append(clf_results['train_time'])
        temp_names.append(clf_results['clf_name'].replace('Classifier', ''))
    clf_lscores.append(temp_scores)
    clf_ltrain_times.append(temp_times)
    clf_lnames.append(temp_names)

print clf_lnames
print clf_lscores
print clf_ltrain_times

#Generate bar chart plots using selected benchmarking results

advant

for j in range(len(clf_lscores)):

    n_groups = len(clf_lscores[j])
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4

    impdata1 = clf_lscores[j]
    impdata2 = clf_ltrain_times[j]

    plt.subplot(2, 3, j+1)

    plt.xlabel('Classifier')
    plt.ylabel('F-Score Test ')
    plt.title('Classifier F-Score on Test Data Set')
    plt.xticks(index, clf_lnames[j])
    
    bar1 = plt.bar(index, impdata1, bar_width, alpha=opacity, color='k')
    for rect in bar1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%f' % height, ha='center', va='top',color='b')
    

    plt.subplot(2, 3, j+4)

    plt.xlabel('Classifier')
    plt.ylabel('Train Times ')
    plt.title('Classifier Training Data Set Times')
    plt.xticks(index, clf_lnames[j])
    
    bar2 = plt.bar(index, impdata2, bar_width, alpha=opacity, color='g')
    for rect in bar2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%f' % height, ha='center', va='top',color='b')
    

plt.tight_layout()
plt.show()
