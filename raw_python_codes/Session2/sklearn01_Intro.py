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
    sklearn.decomposition : Algorithms regarding Dimensionality Reduction

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
    
6. <Utilities>
    sklearn.pipeline: pipeline of (feature engineering -> ML Algorithms -> Prediction)
    
Train data with fit() and Predict data with predict()
"""


# TODO: load iris data, get train data and label(target) data
iris = load_iris()
train_data = iris.data
train_label = iris.target

# TODO: Choose your model
clf = GaussianNB()
# clf = DecisionTreeClassifier()

# TODO: Train
clf.fit(train_data, train_label)

# TODO: predict answers based on train data
pred = clf.predict(train_data)

# TODO: Evaluate your prediction by comparing it with label you used for training
print("Accuracy : ", accuracy_score(train_label, pred))