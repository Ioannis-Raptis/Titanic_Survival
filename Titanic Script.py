# Load the datasets and the required packages.
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
dataset_train = pd.read_csv('data\\train.csv')
dataset_test = pd.read_csv('data\\test.csv')

# View the column names and their types
dataset_train.info()

# View the first five lines
dataset_train.head()

# Check missing values in train data
dataset_train.isnull().sum()

# Convert variable Pclass from numerical to categorical
dataset_train.Pclass = dataset_train.Pclass.astype('category')

# View the Pclass distribution
Pclass_plot = sns.countplot(
    x='Pclass', data=dataset_train, palette="Pastel1_r")

# View the Survived distribution per Sex
dataset_train['Survived'] = dataset_train['Survived'].replace(
    0, 'No')
dataset_train['Survived'] = dataset_train['Survived'].replace(
   1, 'Yes')
sns.countplot(x='Survived', hue='Sex', data=dataset_train)

# View the Sex distribution
survived_count = dataset_train['Sex'].value_counts()
labels = 'Male', 'Female'
plt.pie(x=survived_count, labels=labels, autopct='%1.1f%%',
        colors=['lightcoral', 'lightskyblue'])
plt.show()

# View the Embarked distribution. (We rename the values of the variable).
dataset_train['Embarked'] = dataset_train['Embarked'].replace('C', 'Cherbourg')
dataset_train['Embarked'] = dataset_train['Embarked'].replace(
    'Q', 'Queenstown')
dataset_train['Embarked'] = dataset_train['Embarked'].replace(
    'S', 'Southampton')
sns.countplot(x='Embarked', data=dataset_train)

# View the Age distribution
sns.distplot(dataset_train.Age, bins=20)

# View the Fare distribution
sns.set_style('darkgrid')
sns.distplot(dataset_train.Fare)

# Check for independence between variables sex and survived
# We first create a contigency table
contingency_table = pd.crosstab(dataset_train['Sex'],
                                dataset_train['Survived'], margins=True)

# We then create an array with male / female values
femalecount = contingency_table.iloc[0].values
malecount = contingency_table.iloc[1].values
f_obs = np.array([femalecount, malecount])

# And then we input that array into the x^2 test function
scipy.stats.chi2_contingency(f_obs)[0:3]

# Based on our results we reject the null hypothesis. The variables
# are not independent.

# Subset the Dataset to exclude lines with no value for Age and Embarked
dataset_train_noNULLS = dataset_train[dataset_train['Age'].notnull(
) & dataset_train['Embarked'].notnull()]

# Create the independent and dependent variables
# Get the dummies to convert categorical variables to numerical
# Drop one dummy for each variable to avoid the dummy variable trap
cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
x_train = pd.get_dummies(dataset_train_noNULLS[cols])

x_train.drop('Pclass_3', axis=1, inplace=True)
x_train.drop('Embarked_Southampton', axis=1, inplace=True)
x_train.drop('Sex_male', axis=1, inplace=True)

y_train = dataset_train_noNULLS['Survived']

# Build a logreg and compute the feature importances
model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)
model.coef_

# Try the RFE method
# Create the RFE model and select number of attributes
# We checked the appropriate number of attributes through the confusion matrix
rfe = RFE(model, 7).fit(x_train, y_train)

# summarize the selection of the attributes
print('Selected features: %s' % list(x_train.columns[rfe.support_]))

# Create a confusion matrix in the form of an array
y_pred = rfe.predict(x_train)
cnf_matrix = metrics.confusion_matrix(y_train, y_pred)
cnf_matrix

# Plot of predicted probabilities of survival vs actual survival
y_pred_prob = rfe.predict_proba(x_train)[:, 1]
plt.scatter(y_pred_prob, y_train, s=10)
plt.xlabel('Predicted Chance Of Survival')
plt.ylabel('Actual Survival')
plt.tight_layout()
plt.show()

print("Accuracy:", metrics.accuracy_score(y_train, y_pred))

#####################################################
# Test our model against dataset_test
# Fill out the missing values of age with the mean
# Fill out the missing values of fare with the median
mean_age = dataset_test.loc[:, 'Age'].mean()
print('The mean age in the test dataset is', round(mean_age, 1))
dataset_test['Age'] = dataset_test['Age'].fillna(round(mean_age, 1))

median_fare = dataset_test.loc[:, 'Fare'].median()
print('The median fare in the test dataset is', round(median_fare, 1))
dataset_test['Fare'] = dataset_test['Fare'].fillna(round(mean_age, 1))

# Format the test dataset as we did with the train set and get the dummies
dataset_test['Embarked'] = dataset_test['Embarked'].replace('C', 'Cherbourg')
dataset_test['Embarked'] = dataset_test['Embarked'].replace(
    'Q', 'Queenstown')
dataset_test['Embarked'] = dataset_test['Embarked'].replace(
    'S', 'Southampton')

dataset_test.Pclass = dataset_test.Pclass.astype('category')
x_test = pd.get_dummies(dataset_test[cols])

x_test.drop('Pclass_3', axis=1, inplace=True)
x_test.drop('Embarked_Southampton', axis=1, inplace=True)
x_test.drop('Sex_male', axis=1, inplace=True)

# Check for missing values
x_test.isnull().sum()

# Check that the shape of the train and test datasets is the same
x_test.shape
x_train.shape

# Make the predictions of survival in yes / no and in probabilities
y_pred_test = rfe.predict(x_test)

y_pred_prob_test = rfe.predict_proba(x_test)[:, 1]

dataset_actuals = pd.read_csv('data\\test survived actuals.csv')
y_actuals = dataset_actuals['Survived']
y_actuals = y_actuals.replace(
    0, 'No')
y_actuals = y_actuals.replace(
   1, 'Yes')


# Create a confusion matrix for the test-actual datasets
cnf_matrix = metrics.confusion_matrix(y_actuals, y_pred_test)
cnf_matrix

print("Accuracy:", round(metrics.accuracy_score(y_actuals, y_pred_test),3))
