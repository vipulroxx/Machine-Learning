# In this exercise, we'll use the Titanic dataset as before, train two classifiers and
# look at their confusion matrices. Your job is to create a train/test split in the data
# and report the results in the dictionary at the bottom.

import numpy as np
import pandas as pd

# Load the dataset
from sklearn import datasets

X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# TODO: split the data into training and testing sets,
# using the default settings for train_test_split (or test_size = 0.25 if specified).
# Then, train and test the classifiers with your newly split data instead of X and y.
x_label, x_test, y_label, y_test = train_test_split(X,y)

clf1 = DecisionTreeClassifier()
clf1.fit(x_label , y_label)
print "Confusion matrix for this Decision Tree:\n",confusion_matrix(y_test,clf1.predict(x_test))

clf2 = GaussianNB()
clf2.fit(x_label , y_label)
print "GaussianNB confusion matrix:\n",confusion_matrix(y_test,clf2.predict(x_test))

#TODO: store the confusion matrices on the test sets below
confusions = {
 "Naive Bayes": confusion_matrix(y_test,clf1.predict(x_test)),
 "Decision Tree": confusion_matrix(y_test,clf2.predict(x_test))
}
