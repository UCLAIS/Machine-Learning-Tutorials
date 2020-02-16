from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# TODO: Load iris dataset
iris = load_iris()

# TODO: print each iris dataset and check info
print("iris data: \n", iris.data)
print("iris target: \n", iris.target)

print("feature names: \n", iris.feature_names)
print("target names: \n", iris.target_names)

data = pd.DataFrame(iris.data, columns=iris.feature_names)
print(data.head(10), '\n')
print(data.describe(), '\n')
print(data.info())

# TODO: Homework -> Draw meaningful graphs using pandas and matplotlib

# TODO: Load classifier model
clf = DecisionTreeClassifier()

# ----------- Not splitting Data ------------------------------

# TODO: Train without splitting data -> do the same thing as you did in the sklearn01_Intro.py
clf.fit(iris.data, iris.target)

# TODO: predict targets based on your x datasets
pred = clf.predict(iris.data)

# TODO: Evaluate your prediction by comparing it with label you used for training -> same as before
# You must get accuracy 100% Why?
print("Accuracy : {0:.2f}%".format(accuracy_score(iris.target, pred) * 100))

# But do you think this is a correct way to evaluate your accuracy?
# NO. You are testing prediction ability based on what you trained.
# This is why you need train test split.

# ----------- Splitting Data with train and test ---------------

# TODO: Split your data into X_train, X_test, Y_train and Y_test
# TODO: What is optimal rate for the test_size?

"""
Check URL: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

train_test_split(arrays, test_size, train_size, random_state, shuffle, stratify)

arrays : x and y data
test_size : Ratio of Test data (default = 0.25)
train_size : Ratio of Train data (default = 1 - 0.25)
random_state : seed value for shuffle. It is used to seed a new RandomState object. This is to check and validate the data when running the code multiple times.
shuffle : shuffle or not? (default = True)
stratify : will discuss later on (default = None)
"""

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42, shuffle=True)

# TODO: Train your model with your 'train data' not the whole data
clf.fit(X_train, Y_train)

# TODO: Predict targets with your 'test data' not the whole target data
pred = clf.predict(X_test)

# TODO: Evaluate your prediction. Which data should you compare your predicted data with?
print("Accuracy : {0:.2f}%".format(accuracy_score(Y_test, pred) * 100))



