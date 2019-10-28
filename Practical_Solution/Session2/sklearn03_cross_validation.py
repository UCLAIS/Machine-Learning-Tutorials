import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

"""
Check URL: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

Cross Validation is encouraged to use when dataset is not big enough.
Cross Validation is used to avoid overfitting
Overfitting means literally 'data is overly fitted with the data' so that it does not perform well when new data are given.

K-fold Cross Validation is a methodology that makes the K number of data set 
and eventually train and test the data K times iteratively.

KFold(n_splits)

n_splits : the number of folds (splits)
KFold(n_splits=N). split(X)

X : Data
"""


# TODO: Load iris
iris = load_iris()

# TODO: Load model
clf = DecisionTreeClassifier()

# TODO: How many folds do you want?
n_iter = 1      # this var is not needed. made it just to keep track of the number of iteration
kfold = KFold(n_splits = 5)
cv_accuracy = []

# type of idx => numpy ndarray
for train_idx, test_idx in kfold.split(iris.data):
    print(train_idx)
    print(train_idx.shape)
    print(test_idx)
    X_train, X_test = iris.data[train_idx], iris.data[test_idx]
    y_train, y_test = iris.target[train_idx], iris.target[test_idx]

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    accuracy = np.round(accuracy_score(y_test, pred), 3)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    print("Iteration : {}, Cross-Validation Accuracy : {}, Size of Train Data : {}, Size of Test Data : {}"
          .format(n_iter, accuracy, train_size, test_size))

    n_iter += 1

    cv_accuracy.append(accuracy)

print("Average accuracy : ", np.mean(cv_accuracy))


