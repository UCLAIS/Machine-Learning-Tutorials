import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

"""
Helpful URL to understand this concept:
1. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

<code from sciket-learn doc.>
import numpy as np
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])  -->
y = np.array([0, 0, 1, 1])                      --> Here you can see that the distribution of the data is unbalanced, where normal KFold is not desirable
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

for train_index, test_index in skf.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]


2. https://www.quora.com/What-is-difference-between-k-fold-and-stratified-k-fold-cross-validations
3. https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation


Stratified KFold is encouraged to use when the distribution of the given data is unbalanced,
    where if you just do KFold it creates high bias to your model.


StratifiedKFold(n_splits)
n_splits : the number of splits i.e. folds

StratifiedKFold(n_splits=5).split(X, Y)
X : Data
Y : label
"""

# load iris
iris = load_iris()

# Let's check the distribution of data by making the data as DF
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target

# Classification model
dt_clf = DecisionTreeClassifier()

# Iris data comprise of 3 type of classes
# Therefore it is good to split into 3
n_iter = 0      # this var is not needed. just for tracking the iteration
skf = StratifiedKFold(n_splits=3)
avg_acc = []

for train_idx, test_idx in skf.split(iris.data, iris.target):

    n_iter += 1

    # TODO: split it and assign data to variables
    X_train, X_test = iris.data[train_idx], iris.data[test_idx]
    y_train, y_test = iris.target[train_idx], iris.target[test_idx]

    print("Iteration :", n_iter)
    print("--------------------")

    '''
    Refer to this site (about Series.value_counts()): 
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html
    '''

    print("Check distribution of train data : \n",
          iris_df['label'].iloc[train_idx].value_counts()
          )
    print("--------------------")
    print("Check distribution of test data : \n",
          iris_df['label'].iloc[test_idx].value_counts()
          )
    print("--------------------")

    # train your model with train data
    dt_clf.fit(X_train, y_train)

    # make your model predict data with test data
    pred = dt_clf.predict(X_test)

    # TODO: Evaluate your model
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    print("Iteration : {}, Accuracy : {}%, Size of Train data : {}, Size of Test data : {}\n"
          .format(n_iter, accuracy * 100, train_size, test_size))

    avg_acc.append(accuracy)

print("Average accuracy : {0:.4f}%".format(np.mean(avg_acc) * 100))
