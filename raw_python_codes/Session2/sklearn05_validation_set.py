import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold

"""
Train, Validation, Test

Check URL: https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7

You split training set into train and validation sets
And check its performance with test datasets

"""

iris = load_iris()


# TODO: train_test_split(), choose good test_size
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# TODO: declare kfold
# kfold = KFold(n_splits=3)
STK = StratifiedKFold(n_splits=3)

# TODO: Split X_train into train and validation set
for train_idx, vali_idx in STK.split(X_train, y_train):
    X_train, X_vali = iris.data[train_idx], iris.data[vali_idx]
    y_train, y_vali = iris.target[train_idx], iris.target[vali_idx]
