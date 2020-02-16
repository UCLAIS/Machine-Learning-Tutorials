import numpy as np
from sklearn.linear_model import LinearRegression
import csv
import matplotlib.pyplot as plt

csvReader = csv.reader(open("../../notebooks/data/Advertisement.csv"))
# skip the header
next(csvReader)

x = []
y = []

for row in csvReader:
    x_i = [float(row[1]), float(row[2]), float(row[3])]
    x.append(x_i)

    y_i = float(row[4])
    y.append(y_i)

X = np.array(x)
Y = np.array(y)

########################
# Train the model here #
########################
# TODO: load linear regression model
lrmodel = LinearRegression()
# TODO: fit the model
lrmodel.fit(X, Y)
Y_pred = lrmodel.predict(X)

print("X size: ", X.size)
print("Y_pred size: ", Y_pred.size)

# TODO: Think. Can you draw a graph using plt.plot(X, Y_pred)?

beta_0 = lrmodel.coef_[0]
beta_1 = lrmodel.coef_[1]
beta_2 = lrmodel.coef_[2]
beta_3 = lrmodel.intercept_

print("beta_0: %f" % beta_0)
print("beta_1: %f" % beta_1)
print("beta_2: %f" % beta_2)
print("beta_3: %f" % beta_3)


# infer an expected sales based on given values
def expected_sales(fb, insta, twitter, beta_0, beta_1, beta_2, beta_3):
    # Multiple Linear Regression Model
    sales = (beta_0 * fb) + (beta_1 * insta) + (beta_2 * twitter) + beta_3
    return sales


# Sales prediction, Given that fb:10, insta:12, twitter:3
expectation = expected_sales(100, 80, 40, beta_0, beta_1, beta_2, beta_3)
print("Expected Sales: {}".format(expectation))
