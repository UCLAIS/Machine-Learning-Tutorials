from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

"""
Why do we use sklearn??

1. <Example Datasets>
    sklearn.datasets : Provides example datasets

2. <Feature Engineering>
    sklearn.preprocessing : Variable functions as to data preprocessing
    sklearn.feature_selection : Help selecting primary components in datasets
    sklearn.feature_extraction : Vectorised feature extraction
    sklearn.decomposition : Algorithms as to Dimensionality Reduction

3. <Data split and Parameter Tuning>
    sklearn.model_selection : 'Train Test Split' for cross validation, Parameter tuning with GridSearch

4. <Evaluation>
    sklearn.metrics : accuracy score, ROC curve, F1 score, etc.

5. <ML Algorithms>
    sklearn.ensemble : Ensemble, etc.
    sklearn.linear_model : Linear Regression, Logistic Regression, etc.
    sklearn.naive_bayes : Gaussian Naive Bayes classification, etc.
    sklearn.neighbors : Nearest Centroid classification, etc.
    sklearn.svm : Support Vector Machine
    sklearn.tree : DecisionTreeClassifier, etc.
    sklearn.cluster : Clustering (Unsupervised Learning)
"""


# TODO: load iris data, get train data and label(target) data
iris = None
train_data = None
train_label = None

# TODO: Choose your model
clf = None

# TODO: Train

# TODO: predict answers based on train data
pred = None

# TODO: Evaluate your prediction by comparing it with label you used for training
