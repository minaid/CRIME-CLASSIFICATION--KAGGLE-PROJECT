import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M')

#finding missing values, nulls and type of the values
training = pd.read_csv('training.csv', parse_dates=['Dates'], date_parser=dateparse)
testing = pd.read_csv('testing.csv', header = 0)

dateds = training.Dates.value_counts()
print ((dateds), '\n')

print((training.info()), '\n')
print((pd.isnull(training).sum()), '\n')

print((testing.info()), '\n')
print((pd.isnull(testing).sum()), '\n')

#From the results,training file has 2500 rows & 9 columns, while testing has 500 rows & 9 columns.
#None of them has missing values or nulls in any of their cells.

#The result of the codes below helps us see how many times each value appears: Category, DayOfWeek
#PdDistrict and Resolution.
number_of_crimes = training.Category.value_counts()
days_of_crimes = training.DayOfWeek.value_counts()
districts_of_crimes = training.PdDistrict.value_counts()
resolutions_of_crimes = training.Resolution.value_counts()
print ((number_of_crimes), '\n')
print ((days_of_crimes), '\n')
print ((districts_of_crimes), '\n')
print ((resolutions_of_crimes), '\n')


#Replacing the Category:
categ = training["Category"].unique()
data_dict = {}
count = 1
for data in categ:
    data_dict[data] = count
    count+=1
    
training["Category"] = training["Category"].replace(data_dict)
testing["Category"] = testing["Category"].replace(data_dict)

#Replacing the Descript:
categ = training["Descript"].unique()
data_dict = {}
count = 1
for data in categ:
    data_dict[data] = count
    count+=1
    
training["Descript"] = training["Descript"].replace(data_dict)
testing["Descript"] = testing["Descript"].replace(data_dict)

#Replacing the DayOfWeek:
data_week_dict = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7
    }

training["DayOfWeek"] = training["DayOfWeek"].replace(data_week_dict)
testing["DayOfWeek"] = testing["DayOfWeek"].replace(data_week_dict)

#Replacing the PdDistrict:
Pd_dict = {
    "NORTHERN": 1,
    "PARK": 2,
    "INGLESIDE": 3,
    "BAYVIEW": 4,
    "RICHMOND": 5,
    "CENTRAL": 6,
    "TARAVAL": 7,
    "TENDERLOIN": 8,
    "MISSION": 9,
    "SOUTHERN": 10
    }
    
training["PdDistrict"] = training["PdDistrict"].replace(Pd_dict)
testing["PdDistrict"] = testing["PdDistrict"].replace(Pd_dict)

#Replacing the Resolution:
res_dict = {
    "ARREST_BOOKED": 1,
    "ARREST_CITED": 2,
    "PSYCHOPATHIC CASE": 3,
    "JUVENILE BOOKED": 4,
    "NONE": 5,
    "UNFOUNDED": 6,
    "EXCEPTIONAL CLEARANCE": 7,
    "LOCATED": 8,
    'CLEARED-CONTACT JUVENILE FOR MORE INFO': 9
    }
    
training["Resolution"] = training["Resolution"].replace(res_dict)
testing["Resolution"] = testing["Resolution"].replace(res_dict)

print((training[['Resolution', 'Category']].groupby(['Resolution'], as_index=False).mean().sort_values(by='Category', ascending=False)), '\n')

print((training.head()), '\n')
print((testing.head()), '\n')

training = training.drop(["Dates", "Address", "X", "Y"], axis=1)
testing = testing.drop(["Dates", "Address", "X", "Y"], axis=1)

print((training.head()), '\n')
print((testing.head()), '\n')

columns_train = training.columns
print((columns_train), '\n')
columns_test = testing.columns
print((columns_test), '\n')

print((training.describe()), '\n')
print((testing.describe()), '\n')

import pylab as pl


training.drop("Category" ,axis=1).hist(bins=30, figsize=(9,9))
pl.suptitle("Histogram for each numeric input variable")
plt.savefig("Category")
plt.show()

#Have a look at correlations
corr = training.corr()
print((corr["Category"]), '\n')
print((corr["Descript"]), '\n')
print((corr["DayOfWeek"]), '\n')
print((corr["PdDistrict"]), '\n')
print((corr["Resolution"]), '\n')

import seaborn as sns

sns.set(font_scale=1)

sns.heatmap(training[['Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution']].corr(),cmap="BrBG_r",annot=True,linecolor='white')
plt.title('Correlation between Features ')
plt.show()

#Calculate the skew
skew = training.skew()
print((skew), '\n')

#Let's use different kind of classifiers to find the one that fit better at the dataset
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

x=training.drop(['Category'], axis=1)
y=training['Category']

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=0)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
print("Mean accuracy of knn: ", metrics.accuracy_score(predictions, y_test), '\n')

ran = RandomForestClassifier(n_estimators=100, max_features=3)
ran.fit(x_train, y_train)
predictions = ran.predict(x_test)
print("Mean accuracy of Random Forest: ", metrics.accuracy_score(predictions, y_test), '\n')

kfold = model_selection.KFold(n_splits=40, random_state=7)

dec = DecisionTreeClassifier(random_state = 0)
dec.fit(x_train, y_train)
results1 = model_selection.cross_val_score(dec, x, y, cv=kfold)
print("Mean accuracy of Decision Trees:" + str(results1.mean()), '\n')

bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=0)
bag.fit(x_train, y_train)
results2 = model_selection.cross_val_score(bag, x, y, cv=kfold)
print("Mean accuracy of Bagging:" + str(results2.mean()), '\n')

_n_crime_plot = sns.barplot(x=number_of_crimes.index,y=number_of_crimes)
_n_crime_plot.set_xticklabels(number_of_crimes.index,rotation=90)
plt.show()

#seaborn histogram
sns.distplot(training['Category'], hist=True, kde=False, 
            bins=int(30), color = 'green',
            hist_kws={'edgecolor':'red'})
#Add labels
plt.title('Histogram of Category')
plt.show()

_n_crime_plot = sns.barplot(x=days_of_crimes.index,y=days_of_crimes)
_n_crime_plot.set_xticklabels(days_of_crimes.index,rotation=90)
plt.show()

sns.distplot(training['DayOfWeek'], hist=True, kde=False, 
            bins=int(30), color = 'green',
            hist_kws={'edgecolor':'red'})
#Add labels
plt.title('Histogram of DayOfWeek')
plt.show()

_n_crime_plot = sns.barplot(x=districts_of_crimes.index,y=districts_of_crimes)
_n_crime_plot.set_xticklabels(districts_of_crimes.index,rotation=90)
plt.show()

sns.distplot(training['PdDistrict'], hist=True, kde=False, 
            bins=int(30), color = 'green',
            hist_kws={'edgecolor':'red'})
#Add labels
plt.title('Histogram of PdDistrict')
plt.show()


_n_crime_plot = sns.barplot(x=resolutions_of_crimes.index,y=resolutions_of_crimes)
_n_crime_plot.set_xticklabels(resolutions_of_crimes.index,rotation=90)
plt.show()

sns.distplot(training['Resolution'], hist=True, kde=False, 
            bins=int(30), color = 'green',
            hist_kws={'edgecolor':'red'})
#Add labels
plt.title('Histogram of Resolution')
plt.show()

pareto_crime = number_of_crimes / sum(number_of_crimes)
pareto_crime = pareto_crime.cumsum()
_pareto_crime_plot = sns.tsplot(data=pareto_crime)
_pareto_crime_plot.set_xticklabels(pareto_crime.index,rotation=90)
_pareto_crime_plot.set_xticks(np.arange(len(pareto_crime)))
plt.show()                             

Main_Crime_Categories = list(pareto_crime[0:5].index)
print("Most common categories: ")
print((Main_Crime_Categories), '\n')
print(("that make up to {:.2%} of the crimes.".format(pareto_crime[8])), '\n')

output = pd.DataFrame(dec.predict(x_test))
output.index+=1
output.to_csv('Submission.csv')

