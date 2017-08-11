
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load the Census dataset
data = pd.read_csv("census.csv")
# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)
sub_features = ['education_level', 'occupation', 'marital-status', 'race', 'sex']


if True:

    #create figure
    plt.figure(1, figsize=(20, 10))
    plot_num = 1
    nrows = 3
    ncols = 4
    for feat_choice in sub_features:
        groups = features_raw[feat_choice].unique()
        gelems = len(groups)
        less_than_50 = [0] * gelems
        more_than_50 = [0] * gelems
        for ocn in range(len(groups)):
            for row in range(len(features_raw)):
                if features_raw[feat_choice][row].strip() == groups[ocn].strip() and income_raw[row].strip() == '>50K':
                    more_than_50[ocn] += 1
                elif features_raw[feat_choice][row].strip() == groups[ocn].strip() and income_raw[row].strip() == '<=50K':
                    less_than_50[ocn] += 1

        # create subplots
        plt.subplot(nrows, ncols, plot_num)
        index = np.arange(gelems)
        bar_width = 0.35
        opacity = 0.8
        rects1 = plt.bar(index, less_than_50, bar_width, alpha=opacity, color='k', label='<=50K')
        rects2 = plt.bar(index + bar_width, more_than_50, bar_width, alpha=opacity, color='g', label='>50K')
        plt.ylabel('No. Records')
        plt.title('Income by {}'.format(feat_choice))
        plt.xticks(index + bar_width / 2.0, groups, rotation='vertical')
        plt.legend(frameon=False, loc='upper right', fontsize='small')
        plot_num += 1

    # create extra subplot of income by age

    age_groups = ['10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']
    n_groups_age = len(age_groups)
    less_than_50_age = [0] * n_groups_age
    more_than_50_age = [0] * n_groups_age
    d_age = 10
    offset = 0
    for agn in range(len(age_groups)):
        if agn == len(age_groups) - 1: offset = 1
        for row in range(len(features_raw)):
            if d_age <= features_raw['age'][row] < d_age + 10 + offset and income_raw[row].strip() == '>50K':
                more_than_50_age[agn] += 1
            elif d_age <= features_raw['age'][row] < d_age + 10 + offset and income_raw[row].strip() == '<=50K':
                less_than_50_age[agn] += 1
        d_age += 10

    plt.subplot(nrows, ncols, plot_num)
    index = np.arange(n_groups_age)
    bar_width = 0.35
    opacity = 0.8
    rects1 = plt.bar(index, less_than_50_age, bar_width, alpha=opacity, color='k', label='<=50K')
    rects2 = plt.bar(index + bar_width, more_than_50_age, bar_width, alpha=opacity, color='g', label='>50K')
    plt.xlabel('Age')
    plt.ylabel('No. Records')
    plt.title('Income by Age')
    plt.xticks(index + bar_width / 2.0, age_groups)
    plt.legend(frameon=False, loc='upper right', fontsize='small')

    # create extra subplot of income by capital change

    cap_gain_groups = ['0-10K', '10-20K', '20-30K', '30-40K', '40-50K', '50-60K', '60-70K', '70-80K', '80-90K', '90-100K']
    cap_loss_groups = ['0-500', '500-1K', '1-1.5K', '1.5-2K', '2-2.5K', '2.5-3K', '3-3.5K', '3.5-4K','4-4.5K', '4.5-5K']
    gr_set = [cap_gain_groups, cap_loss_groups]
    cap_feat = ['capital-gain', 'capital-loss']


    for m, feat in enumerate(cap_feat):
        n_groups_cap = len(gr_set[m])
        less_than_50 = [0] * n_groups_cap
        more_than_50 = [0] * n_groups_cap
        d_cap = 0
        steps = [10000, 500]
        offset = 0
        for agn in range(len(gr_set[m])):
            if agn == len(gr_set[m]) - 1: offset = 100
            for row in range(len(features_raw)):
                if d_cap <= features_raw[feat][row] < d_cap + steps[m] + offset and income_raw[row].strip() == '>50K':
                    more_than_50[agn] += 1
                elif d_cap <= features_raw[feat][row] < d_cap + steps[m] + offset and income_raw[row].strip() == '<=50K':
                    less_than_50[agn] += 1
            d_cap += 10000

        # create subplots
        plt.subplot(nrows, ncols, plot_num+1)
        index = np.arange(n_groups_cap)
        bar_width = 0.35
        opacity = 0.8
        rects1 = plt.bar(index, less_than_50, bar_width, alpha=opacity, color='k', label='<=50K')
        rects2 = plt.bar(index + bar_width, more_than_50, bar_width, alpha=opacity, color='g', label='>50K')
        plt.ylabel('No. Records')
        plt.title('Income by {}'.format(feat))
        plt.xticks(index + bar_width / 2.0, gr_set[m], rotation='vertical')
        plt.legend(frameon=False, loc='upper right', fontsize='small')
        plot_num += 1



    plt.tight_layout()
    plt.show()




#--------------------------6- BAR PLOT: Income by Marital Status -------------------------------------

if False:

    mstat_groups = features_raw['marital-status'].unique()
    n_groups_mstat = len(mstat_groups)

    less_than_50_mst = [0] * n_groups_mstat
    more_than_50_mst = [0] * n_groups_mstat

    for ocn in range(len(mstat_groups)):
        for row in range(len(features_raw)):
            if features_raw['marital-status'][row].strip() == mstat_groups[ocn].strip() and income_raw[row].strip() == '>50K':
                more_than_50_mst[ocn] += 1
            elif features_raw['marital-status'][row].strip() == mstat_groups[ocn].strip() and income_raw[row].strip() == '<=50K':
                less_than_50_mst[ocn] += 1

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups_mstat)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, less_than_50_mst, bar_width,
                     alpha=opacity,
                     color='k',
                     label='<=50K')

    rects2 = plt.bar(index + bar_width, more_than_50_mst, bar_width,
                     alpha=opacity,
                     color='g',
                     label='>50K')

    #plt.xlabel('Marital Status')
    plt.ylabel('No. Individuals')
    plt.title('Income by Race')
    plt.xticks(index + bar_width / 2.0, mstat_groups, rotation='vertical')
    plt.legend()

    plt.tight_layout()
    plt.show()




#--------------------------5- BAR PLOT: Income by Race -------------------------------------

if False:

    race_groups = features_raw['race'].unique()
    n_groups_race = len(race_groups)

    less_than_50_race = [0] * n_groups_race
    more_than_50_race = [0] * n_groups_race

    for ocn in range(len(race_groups)):
        for row in range(len(features_raw)):
            if features_raw['race'][row].strip() == race_groups[ocn].strip() and income_raw[row].strip() == '>50K':
                more_than_50_race[ocn] += 1
            elif features_raw['race'][row].strip() == race_groups[ocn].strip() and income_raw[row].strip() == '<=50K':
                less_than_50_race[ocn] += 1

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups_race)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, less_than_50_race, bar_width,
                     alpha=opacity,
                     color='k',
                     label='<=50K')

    rects2 = plt.bar(index + bar_width, more_than_50_race, bar_width,
                     alpha=opacity,
                     color='g',
                     label='>50K')

    #plt.xlabel('Race')
    plt.ylabel('No. Individuals')
    plt.title('Income by Race')
    plt.xticks(index + bar_width / 2.0, race_groups, rotation=10)
    plt.legend()

    plt.tight_layout()
    plt.show()


#--------------------------4- BAR PLOT: Income by Education Level -------------------------------------

if False:

    edu_groups = features_raw['education_level'].unique()
    n_groups_edu = len(edu_groups)

    less_than_50_edu = [0] * n_groups_edu
    more_than_50_edu = [0] * n_groups_edu

    for ocn in range(len(edu_groups)):
        for row in range(len(features_raw)):
            if features_raw['education_level'][row].strip() == edu_groups[ocn].strip() and income_raw[row].strip() == '>50K':
                more_than_50_edu[ocn] += 1
            elif features_raw['education_level'][row].strip() == edu_groups[ocn].strip() and income_raw[row].strip() == '<=50K':
                less_than_50_edu[ocn] += 1

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups_edu)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, less_than_50_edu, bar_width,
                     alpha=opacity,
                     color='k',
                     label='<=50K')

    rects2 = plt.bar(index + bar_width, more_than_50_edu, bar_width,
                     alpha=opacity,
                     color='g',
                     label='>50K')

    plt.xlabel('Education Level')
    plt.ylabel('No. Individuals')
    plt.title('Income by Education Level')
    plt.xticks(index + bar_width / 2.0, edu_groups, rotation='vertical')
    plt.legend()

    plt.tight_layout()
    plt.show()


#--------------------------3- BAR PLOT: Income by Occupation -------------------------------------

if False:

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
    plt.ylabel('No. Individuals')
    plt.title('Income by Occupation')
    plt.xticks(index + bar_width / 2.0, occupations, rotation='vertical')
    plt.legend()

    plt.tight_layout()
    plt.show()

#--------------------------2- BAR PLOT: Income by Gender -------------------------------------

if False:


    sex_groups = features_raw['sex'].unique()
    n_groups_sex = len(sex_groups)
    less_than_50_sex = [0] * n_groups_sex
    more_than_50_sex = [0] * n_groups_sex

    for sxn in range(len(sex_groups)):
        for row in range(len(features_raw)):

            if features_raw['sex'][row].strip() == sex_groups[sxn].strip() and income_raw[row].strip() == '>50K':
                more_than_50_sex[sxn] += 1
            elif features_raw['sex'][row].strip() == sex_groups[sxn].strip() and income_raw[row].strip() == '<=50K':
                less_than_50_sex[sxn] += 1

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
    plt.ylabel('No. Individuals')
    plt.title('Income by Gender')
    plt.xticks(index + bar_width/2.0, sex_groups)
    plt.legend()

    plt.tight_layout()
    plt.show()

#--------------------------1- BAR PLOT: Income by Age -------------------------------------

if False:

    age_groups = ['10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']
    n_groups_age = len(age_groups)

    less_than_50_age = [0] * n_groups_age
    more_than_50_age = [0] * n_groups_age

    d_age = 10
    offset = 0
    for agn in range(len(age_groups)):

        if agn == len(age_groups)-1 : offset = 1

        for row in range(len(features_raw)):

            if d_age <= features_raw['age'][row] < d_age + 10 + offset and income_raw[row].strip() == '>50K':
                more_than_50_age[agn] += 1
            elif d_age <= features_raw['age'][row] < d_age + 10 + offset and income_raw[row].strip() == '<=50K':
                less_than_50_age[agn] += 1

        d_age += 10

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups_age)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, less_than_50_age, bar_width, alpha=opacity, color='k', label='<=50K')
    rects2 = plt.bar(index + bar_width, more_than_50_age, bar_width, alpha=opacity, color='g', label='>50K')

    plt.xlabel('Age')
    plt.ylabel('No. Individuals')
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



