###################################################
# Churn Prediction
###################################################

# Business Problem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro, mannwhitneyu, spearmanr, pearsonr,chi2_contingency
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)


# Data Overwiew
data = pd.read_csv('notebook/data/Churn_Modelling.csv')
data.head()

data.info()
# #   Column           Non-Null Count  Dtype
# ---  ------           --------------  -----
#  0   RowNumber        10000 non-null  int64
#  1   CustomerId       10000 non-null  int64
#  2   Surname          10000 non-null  object
#  3   CreditScore      10000 non-null  int64
#  4   Geography        10000 non-null  object
#  5   Gender           10000 non-null  object
#  6   Age              10000 non-null  int64
#  7   Tenure           10000 non-null  int64
#  8   Balance          10000 non-null  float64
#  9   NumOfProducts    10000 non-null  int64
#  10  HasCrCard        10000 non-null  int64
#  11  IsActiveMember   10000 non-null  int64
#  12  EstimatedSalary  10000 non-null  float64
#  13  Exited           10000 non-null  int64
# dtypes: float64(2), int64(9), object(3)


data.duplicated().any() # There is no duplicated row.
data.CustomerId.nunique() # 10000 unique customer.
non_important_columns = ['RowNumber', 'CustomerId', 'Surname'] # Unimportant columns
data.drop(non_important_columns, axis=1, inplace=True)

numerical_columns = [col for col in data.columns if data[col].dtype != 'object' and data[col].nunique()>2]
categorical_columns = [col for col in data.columns
                       if data[col].dtype == 'object' or data[col].nunique()==2 and col != 'Exited']

len(numerical_columns) + len(categorical_columns) == len(data.columns)-1 # True

###################################################
# Exploratory Data Aanalysis
###################################################

################################## Target Variable: Exited ##################################
churn_counts = data['Exited'].value_counts()
plt.figure(figsize=(8, 8))
colors = ['#66c2a5', '#fc8d62']

plt.pie(churn_counts,
        labels=['Non Churn', 'Churn'],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=(0.1, 0),
        shadow=True)

plt.title('Proportion of Customers', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.show()

# Comments:
# 20.4% of customers have churned while 79.6% have not. This also indicates a potential imbalance issue.

################################## Numerical Columns ##################################
# 1: Descriptive Statistics

data[numerical_columns].describe().T

#        CreditScore        Age     Tenure     Balance  NumOfProducts  \
# count   10000.0000 10000.0000 10000.0000  10000.0000     10000.0000
# mean      650.5288    38.9218     5.0128  76485.8893         1.5302
# std        96.6533    10.4878     2.8922  62397.4052         0.5817
# min       350.0000    18.0000     0.0000      0.0000         1.0000
# 25%       584.0000    32.0000     3.0000      0.0000         1.0000
# 50%       652.0000    37.0000     5.0000  97198.5400         1.0000
# 75%       718.0000    44.0000     7.0000 127644.2400         2.0000
# max       850.0000    92.0000    10.0000 250898.0900         4.0000
#        HasCrCard  IsActiveMember  EstimatedSalary
# count 10000.0000      10000.0000       10000.0000
# mean      0.7055          0.5151      100090.2399
# std       0.4558          0.4998       57510.4928
# min       0.0000          0.0000          11.5800
# 25%       0.0000          0.0000       51002.1100
# 50%       1.0000          1.0000      100193.9150
# 75%       1.0000          1.0000      149388.2475
# max       1.0000          1.0000      199992.4800

# 2: Distribution and Outlier:
def distribution_outlier_plot(dataframe, col):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.hist(dataframe[col], bins=100, density=False, alpha=0.7, color='#66c2a5', edgecolor='black')
    ax1.set_xlabel(f'{col}', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)

    ax2.boxplot(dataframe[col], patch_artist=True, boxprops=dict(facecolor='#fc8d62', color='black'),
                medianprops=dict(color='yellow'), whiskerprops=dict(color='black'))
    ax2.set_xlabel(f'{col}', fontsize=14)

    fig.tight_layout()
    plt.show()

for col in numerical_columns:
    distribution_outlier_plot(data, col)

# 3: Mannwhitneyu Test

# Assumption Test

def normality(dataframe, col):
    """
    Perform the Shapiro-Wilk test for normality.
    H0 hypothesis: Normal distribution
    """
    stat, pval = shapiro(dataframe[col])
    if pval < 0.05:
        print(f'{col} has not normal distribution: {pval:.3f}')
    else:
        print(f'{col} has normal distribution: {pval:.3f}')


for col in numerical_columns:
    normality(data, col)

# CreditScore has not normal distribution: 0.000
# Age has not normal distribution: 0.000
# Tenure has not normal distribution: 0.000
# Balance has not normal distribution: 0.000
# NumOfProducts has not normal distribution: 0.000
# EstimatedSalary has not normal distribution: 0.000

def mannwhitneyu_test(dataframe, col, target):
    """
    Perform the Mann-Whitney U rank test on two independent samples.
    """
    group_a = dataframe[dataframe[target] == 0][col]
    group_b = dataframe[dataframe[target]==1][col]
    stat, pval = mannwhitneyu(group_a, group_b)
    if pval < 0.05:
        print(f'{col} has statistically significant effect on {target}: pval - {pval:.3f}')
    else:
        print(f'{col} has not statistically significant on {target}: pval - {pval:.3f}')

for col in numerical_columns:
    mannwhitneyu_test(data, col, 'Exited')

# CreditScore has statistically significant effect on Exited: pval - 0.020
# Age has statistically significant effect on Exited: pval - 0.000
# Tenure has not statistically significant on Exited: pval - 0.162
# Balance has statistically significant effect on Exited: pval - 0.000
# NumOfProducts has statistically significant effect on Exited: pval - 0.000
# EstimatedSalary has not statistically significant on Exited: pval - 0.227

def column_target(dataframe, col, target):
    print(dataframe.groupby(target).agg({col: ['mean', 'median']}))

for col in numerical_columns:
    column_target(data, col, 'Exited')
#        CreditScore
#               mean   median
# Exited
# 0         651.8532 653.0000
# 1         645.3515 646.0000
#            Age
#           mean  median
# Exited
# 0      37.4084 36.0000
# 1      44.8380 45.0000
#        Tenure
#          mean median
# Exited
# 0      5.0333 5.0000
# 1      4.9327 5.0000
#           Balance
#              mean      median
# Exited
# 0      72745.2968  92072.6800
# 1      91108.5393 109349.2900
#        NumOfProducts
#                 mean median
# Exited
# 0             1.5443 2.0000
# 1             1.4752 1.0000
#        EstimatedSalary
#                   mean      median
# Exited
# 0           99738.3918  99645.0400
# 1          101465.6775 102460.8400

# 4: Correlation analysis

def correlation_test(dataframe, col, target, method='spearmanr'):
    if method == 'spearmanr':
        stat, pval = spearmanr(dataframe[[target, col]])
        if pval < 0.05:
            print(f'{col} has statistically significant correlation with {target}: pval- {pval:.4f}')
        else:
            print(f'{col} has not statistically significant correlation with {target}: pval - {pval:.4f}')
    else:
        stat, pval = pearsonr(dataframe[[target, col]])
        if pval < 0.05:
            print(f'{col} has statistically significant correlation with {target}: pval- {pval:.4f}')
        else:
            print(f'{col} has not statistically significant correlation with {target}: pval - {pval:.4f}')

for col in numerical_columns:
    correlation_test(data, col, 'Exited', method='spearmanr')

# CreditScore has statistically significant correlation with Exited: pval- 0.0199
# Age has statistically significant correlation with Exited: pval- 0.0000
# Tenure has not statistically significant correlation with Exited: pval - 0.1622
# Balance has statistically significant correlation with Exited: pval- 0.0000
# NumOfProducts has statistically significant correlation with Exited: pval- 0.0000
# EstimatedSalary has not statistically significant correlation with Exited: pval - 0.2271


################################## Categorical Columns ##################################

for col in categorical_columns:
    print(f'{data[col].value_counts()}')

# Geography
# France     5014
# Germany    2509
# Spain      2477

# Gender
# Male      5457
# Female    4543

# HasCrCard
# 1    7055
# 0    2945

# IsActiveMember
# 1    5151
# 0    4849

def proportion_cat_target(dataframe, col, target):
    proportions = dataframe.groupby([col, target])[target].count() / data.groupby([col])[target].count()
    proportions = proportions.unstack()
    print(proportions)

for col in categorical_columns:
    proportion_cat_target(data, col, 'Exited')

# Exited         0      1
# Geography
# France    0.8385 0.1615
# Germany   0.6756 0.3244
# Spain     0.8333 0.1667

# Exited      0      1
# Gender
# Female 0.7493 0.2507
# Male   0.8354 0.1646

# Exited         0      1
# HasCrCard
# 0         0.7919 0.2081
# 1         0.7982 0.2018

# Exited              0      1
# IsActiveMember
# 0              0.7315 0.2685
# 1              0.8573 0.1427

# Comment: The proportion of customers who churned is highest in Germany (32.4%) compared to France (16.2%)
# and Spain (16.7%).

# 2: Chi2 Test:

def Chi2_test(dataframe, col, target):
    """
    Chi-square test of independence of variables in a contingency table

    """
    contingency_table = pd.crosstab(data[target], data[col])
    chi2, pval, dof, ex = chi2_contingency(contingency_table)
    if pval < 0.05:
        print(f'{col}: statistically significant effect on {target}: pval - {pval:.3f}')
    else:
        print(f'{col}: not statistically significant effect on {target}: pval - {pval:.3f}')

for col in categorical_columns:
    Chi2_test(data, col, 'Exited')

