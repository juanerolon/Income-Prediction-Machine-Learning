
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# Load the Census dataset
data = pd.read_csv("census.csv")

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)


print ""

print list(features_raw.columns)

print ""
print ""



#--------------------------3- BAR PLOT: Income by Occupation -------------------------------------

occupations = features_raw['occupation'].unique()
n_groups_occ = len(occupations)

less_than_50_occ = [0] * n_groups_occ
more_than_50_occ = [0] * n_groups_occ

for ocn in range(len(occupations)):
    for row in range(len(features_raw)):
        if features_raw['occupation'][row].strip() == occupations[ocn].strip() and income_raw[row].strip() == '>50K':
            more_than_50_occ[ocn] += 1
        elif features_raw['occupation'][row].strip() == occupations[ocn].strip() and income_raw[row].strip() == '<=50K':
            less_than_50_occ[ocn] += 1

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups_occ)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, less_than_50_occ, bar_width,
                 alpha=opacity,
                 color='k',
                 label='<=50K')

rects2 = plt.bar(index + bar_width, more_than_50_occ, bar_width,
                 alpha=opacity,
                 color='g',
                 label='>50K')

plt.xlabel('Occupation')
plt.ylabel('Income')
plt.title('Income by Occupation')
plt.xticks(index + bar_width / 2.0, occupations, rotation='vertical')
plt.legend()

plt.tight_layout()
plt.show()






#--------------------------2- BAR PLOT: Income by Gender -------------------------------------

if False:


    sex_groups = ['Male', 'Female']
    n_groups_sex = len(sex_groups)
    less_than_50_sex = [0] * n_groups_sex
    more_than_50_sex = [0] * n_groups_sex


    for row in range(len(features_raw)):

        if features_raw['sex'][row].strip() == 'Male' and income_raw[row].strip() == '>50K':
            more_than_50_sex[0] += 1
        elif features_raw['sex'][row].strip() == 'Male' and income_raw[row].strip() == '<=50K':
            less_than_50_sex[0] += 1
        elif features_raw['sex'][row].strip() == 'Female' and income_raw[row].strip() == '>50K':
            more_than_50_sex[1] += 1
        elif features_raw['sex'][row].strip() == 'Female' and income_raw[row].strip() == '<=50K':
            less_than_50_sex[1] += 1
        else:
            print "error"


    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups_sex)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, less_than_50_sex, bar_width,
                     alpha=opacity,
                     color='k',
                     label='<=50K')

    rects2 = plt.bar(index + bar_width, more_than_50_sex, bar_width,
                     alpha=opacity,
                     color='g',
                     label='>50K')

    plt.xlabel('Sex')
    plt.ylabel('Income')
    plt.title('Income by Gender')
    plt.xticks(index + bar_width/2.0, sex_groups)
    plt.legend()

    plt.tight_layout()
    plt.show()





#--------------------------1- BAR PLOT: Income by Age -------------------------------------

if False:


    age_groups = ['17-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']
    n_groups_age = len(age_groups)

    less_than_50_age = [0] * n_groups_age
    more_than_50_age = [0] * n_groups_age

    for row in range(len(features_raw)):

        if 17 <= features_raw['age'][row] < 20 and income_raw[row].strip() == '>50K':
            more_than_50_age[0] += 1
        elif 17 <= features_raw['age'][row] < 20 and income_raw[row].strip() == '<=50K':
            less_than_50_age[0] += 1
        elif 20 <= features_raw['age'][row] < 30 and income_raw[row].strip() == '>50K':
            more_than_50_age[1] += 1
        elif 20 <= features_raw['age'][row] < 30 and income_raw[row].strip() == '<=50K':
            less_than_50_age[1] += 1
        elif 30 <= features_raw['age'][row] < 40 and income_raw[row].strip() == '>50K':
            more_than_50_age[2] += 1
        elif 30 <= features_raw['age'][row] < 40 and income_raw[row].strip() == '<=50K':
            less_than_50_age[2] += 1
        elif 40 <= features_raw['age'][row] < 50 and income_raw[row].strip() == '>50K':
            more_than_50_age[3] += 1
        elif 40 <= features_raw['age'][row] < 50 and income_raw[row].strip() == '<=50K':
            less_than_50_age[3] += 1
        elif 50 <= features_raw['age'][row] < 60 and income_raw[row].strip() == '>50K':
            more_than_50_age[4] += 1
        elif 50 <= features_raw['age'][row] < 60 and income_raw[row].strip() == '<=50K':
            less_than_50_age[4] += 1
        elif 60 <= features_raw['age'][row] < 70 and income_raw[row].strip() == '>50K':
            more_than_50_age[5] += 1
        elif 60 <= features_raw['age'][row] < 70 and income_raw[row].strip() == '<=50K':
            less_than_50_age[5] += 1
        elif 70 <= features_raw['age'][row] < 80 and income_raw[row].strip() == '>50K':
            more_than_50_age[6] += 1
        elif 70 <= features_raw['age'][row] < 80 and income_raw[row].strip() == '<=50K':
            less_than_50_age[6] += 1
        elif 80 <= features_raw['age'][row] < 91 and income_raw[row].strip() == '>50K':
            more_than_50_age[7] += 1
        elif 80 <= features_raw['age'][row] < 91 and income_raw[row].strip() == '<=50K':
            less_than_50_age[7] += 1

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups_age)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, less_than_50_age, bar_width,
                     alpha=opacity,
                     color='k',
                     label='<=50K')

    rects2 = plt.bar(index + bar_width, more_than_50_age, bar_width,
                     alpha=opacity,
                     color='g',
                     label='>50K')

    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title('Income by Age')
    plt.xticks(index + bar_width/2.0, age_groups)
    plt.legend()

    plt.tight_layout()
    plt.show()







##----------------------------------------- DATA TRANSFORMATIONS -----------------------------------------------##

if False:

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



##------------------------------------------------ SCATTER PLOT TEST ---------------------------------------------##

if False:

    features = ['age', 'education-num', 'hours-per-week']

    plt.figure(1, figsize=(16, 5))

    yname = 'Capital Gain'
    fn =1

    #from sklearn.linear_model import LinearRegression
    #reglin = LinearRegression()


    for feat in features:

        plt.subplot(1, 3, fn)
        plt.xlabel(feat)
        plt.ylabel(yname)
        plt.title("Scatter Plot {} vs {}".format(yname,feat))
        #reg_data = data[feat].reshape(-1,1)
        #reglin.fit(reg_data,prices)
        #plt.plot(reg_data,reglin.predict(reg_data),color='brown', linewidth=2)
        plt.scatter(features_final[feat], income, color='turquoise')
        plt.grid(True)

        fn+=1

    plt.tight_layout()
    plt.show()
