import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, train_test_split

"""
Train, Validation, Test

Check URL: https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7

You split training set into train and validation
And check its performance with test datasets

"""

# load mnist digit data
digits = load_digits()


# TODO: train_test_split(), choose good test_size
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

# TODO: declare kfold
kfold = KFold(n_splits=3)

# TODO: Split X_train into train and validation set
for train_idx, vali_idx in kfold.split(X_train):
    X_fold_train, X_fold_vali = digits.data[train_idx], digits.data[vali_idx]
    y_fold_train, y_fold_vali = digits.target[train_idx], digits.target[vali_idx]
