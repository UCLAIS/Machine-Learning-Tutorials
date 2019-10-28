from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# TODO: Load iris dataset
iris = None

# TODO: EDA iris dataset -> see numerical data using pandas before drawing graphs


# TODO: Homework -> Draw meaningful graphs using pandas and matplotlib


# TODO: Load correct model (Should be a classifier)
clf = None

# ----------- Not splitting Data ------------------------------

# TODO: Train without splitting data -> do the same thing as you did in the sklearn01_Intro.py


# TODO: predict targets based on your x datasets
pred = None

# TODO: Evaluate your prediction by comparing it with label you used for training -> same as before
# You should get accuracy above 95% or even 100%


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
random_state : seed value for shuffle
shuffle : shuffle or not? (default = True)
stratify : will discuss later on (default = None)
"""

X_train, X_test, Y_train, Y_test = None

# TODO: Train your model with your 'train data' not the whole data

# TODO: Predict targets with your 'test data' not the whole target data
pred = None

# TODO: Evaluate your prediction. Which data should you compare your predicted data with?



