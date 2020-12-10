import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("training.csv")
test = pd.read_csv("testing.csv")

train_X = data[["Dates", "DayOfWeek","PdDistrict","X","Y"]]
test_X = test[["Dates","DayOfWeek","PdDistrict","X","Y"]]
days = {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":7}
train_X['DayOfWeek'] = train_X['DayOfWeek'].map(days)
test_X['DayOfWeek'] = test_X['DayOfWeek'].map(days)

train_Y = data[["Category"]]
test_Y = test[["Category"]]

def getTime(date):
   tokens = date.split(" ")
   time = tokens[1].split(":")
   return time[0]

def getCategory(time):
   time = int(time)
   category = 0
   if -1<time<8:
       category = 1
   elif 7<time<18:
       category = 2
   elif 17<time<21:
       category = 3
   else:
       category = 1
   return category


for idx, time in enumerate(train_X["Dates"]):
   testTime = getTime(time)
   testCategory = getCategory(testTime)
   train_X["Dates"][idx] = testCategory

for idx, time in enumerate(test_X["Dates"]):
   testTime = getTime(time)
   testCategory = getCategory(testTime)
   test_X["Dates"][idx] = testCategory

import sklearn.preprocessing as preprocessing

def number_encode_features(df):

  result = df.copy()

  encoders = {}

  for column in result.columns:

      if result.dtypes[column] == np.object:

          encoders[column] = preprocessing.LabelEncoder()

          result[column] = encoders[column].fit_transform(result[column])

  return result, encoders

encoded_data_X, encoders1 = number_encode_features(train_X)
encoded_data_Y, encoders2 = number_encode_features(train_Y)

encoded_test_X, encoders3 = number_encode_features(test_X)
encoded_test_Y, encoders4 = number_encode_features(test_Y)


#import seaborn as sns

#sns.heatmap(encoded_data.corr(), square=True)

#plt.show()
#print(encoded_test_X["Address"][200:300])
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(encoded_data_X, encoded_data_Y)

y_pred = clf.predict(encoded_test_X)

import sklearn.metrics as metrics
a1=metrics.accuracy_score(encoded_test_Y, y_pred, normalize=True, sample_weight=None)
print(metrics.accuracy_score(encoded_test_Y, y_pred, normalize=True, sample_weight=None))


import sklearn.linear_model as linear_model

lr = linear_model.LogisticRegression()

lr.fit(encoded_data_X, encoded_data_Y)

y_pred = lr.predict(encoded_test_X)
print(metrics.accuracy_score(encoded_test_Y, y_pred, normalize=True, sample_weight=None))
a2=metrics.accuracy_score(encoded_test_Y, y_pred, normalize=True, sample_weight=None)

if a1>a2:
    print("best method is trees")
else: print("best method is linear reg.")