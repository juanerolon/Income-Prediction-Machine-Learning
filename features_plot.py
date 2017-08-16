
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the Census dataset
data = pd.read_csv("census.csv")
# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)
sub_features = ['education_level', 'occupation', 'marital-status', 'race', 'sex']

#create figure
plt.figure(1, figsize=(16, 16))
plot_num = 1
nrows = 4
ncols = 3
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

    # create subplots for 'education_level', 'occupation', 'marital-status', 'race', 'sex'
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

# create extra subplot for 'age'

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

# create extra subplot for 'capital-gain', 'capital-loss'

cap_gain_groups = ['0-10K', '10-20K', '20-30K', '30-40K', '40-50K', '50-60K', '60-70K', '70-80K', '80-90K', '90-100K']
cap_loss_groups = ['0-500', '500-1K', '1-1.5K', '1.5-2K', '2-2.5K', '2.5-3K', '3-3.5K', '3.5-4K','4-4.5K', '4.5-5K']
weekly_hours = ['0-10h', '10-20h', '20-30h', '30-40h', '40-50h', '50-60h', '60-70h', '70-80h','80-90h', '90-99h']
gr_set = [cap_gain_groups, cap_loss_groups, weekly_hours]
cap_feat = ['capital-gain', 'capital-loss', 'hours-per-week']

for m, feat in enumerate(cap_feat):
    n_groups_cap = len(gr_set[m])
    less_than_50 = [0] * n_groups_cap
    more_than_50 = [0] * n_groups_cap
    d_cap = 0
    steps = [10000, 500,10]
    offset = 0
    for agn in range(len(gr_set[m])):
        if agn == len(gr_set[m]) - 1: offset = 100
        for row in range(len(features_raw)):
            if d_cap <= features_raw[feat][row] < d_cap + steps[m] + offset and income_raw[row].strip() == '>50K':
                more_than_50[agn] += 1
            elif d_cap <= features_raw[feat][row] < d_cap + steps[m] + offset and income_raw[row].strip() == '<=50K':
                less_than_50[agn] += 1
        d_cap += steps[m]

    # create subplots
    plt.subplot(nrows, ncols, plot_num + 1)
    index = np.arange(n_groups_cap)
    bar_width = 0.35
    opacity = 0.8
    rects1 = plt.bar(index, less_than_50, bar_width, alpha=opacity, color='k', label='<=50K')
    rects2 = plt.bar(index + bar_width, more_than_50, bar_width, alpha=opacity, color='g', label='>50K')
    plt.title('Income by {}'.format(feat))
    plt.ylabel('No. Records')
    if feat in ['capital-gain', 'capital-loss']:
        plt.ylim((0, 2000))
    plt.xticks(index + bar_width / 2.0, gr_set[m], rotation='vertical')
    plt.legend(frameon=False, loc='upper right', fontsize='small')
    plot_num += 1

plt.tight_layout()
plt.savefig('feat_selection.pdf')
plt.show()


