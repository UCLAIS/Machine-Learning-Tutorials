from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# TODO: Load iris dataset
iris = load_iris()

# TODO: EDA iris dataset
print(iris.data)
print(iris.target)

print(iris.feature_names)
print(iris.target_names)

data = pd.DataFrame(iris.data, columns=iris.feature_names)
print(data.head(10), '\n')
print(data.describe(), '\n')
print(data.info())

# TODO: Homework -> Draw meaningful graphs using pandas and matplotlib

# TODO: Load correct model (Should be a classifier)
clf = DecisionTreeClassifier()

# ----------- Not splitting Data ------------------------------

# TODO: Train without splitting data -> do the same thing as you did in the sklearn_Intro.py
clf.fit(iris.data, iris.target)

# TODO: predict targets based on your x datasets
pred = clf.predict(iris.data)

# TODO: Evaluate your prediction by comparing it with label you used for training -> same as before
# You should get accuracy above 95% or even 100%
print("Accuracy : {0:.2f}%".format(accuracy_score(iris.target, pred) * 100))

# But do you think this is a correct way to evaluate your accuracy?
# NO. You are testing prediction ability based on what you trained.
# This is why you need train test split.

# ----------- Splitting Data with train and test ---------------

# TODO: Split your data into X_train, X_test, Y_train and Y_test
# TODO: What is optimal rate for the test_size?
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)

# TODO: Train your model with your 'train data' not the whole data
clf.fit(X_train, Y_train)

# TODO: Predict targets with your 'test data' not the whole target data
pred = clf.predict(X_test)

# TODO: Evaluate your prediction. Which data should you compare your predicted data with?
print("Accuracy : {0:.2f}%".format(accuracy_score(Y_test, pred) * 100))



