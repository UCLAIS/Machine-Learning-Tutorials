"""
Let's do Polynomial Regression using scikit-learn

What if our training points are non-linear and resemble curvy cosine or cubic function? We need Polynomial Regression,
 which is often called as Multivariate Regression

PolynomialFeatures(degree): creates Polynomial object
degree: Degree of the polynomial
PolynomialFeatures.fit_transform(x): returns polynomial variables, which are 'x' and 'x to the power of degree'

Check this link out!!
Reference: https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions
"""

from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")


# Random generation of x and y
x = 3 * np.random.rand(100, 3) + 1
print(x[:10])
print("------------")
y = (x ** 2) + x + 2 + 5 * np.random.rand(100, 3)

# TODO: Create PolynomialFeature object(degree=2, include_bias=False)
poly_feat = PolynomialFeatures(degree=2)

# TODO: transform your data (eg: x=[x1, x2] --> poly_x=[1, x1, x2, x1^2, x1x2, x2^2])
poly_x = poly_feat.fit_transform(x)
print(poly_x[0])

# TODO: Load a model. Which model should you load? Correct!! Since we've already transformed the
linear_model = LinearRegression()

# TODO: fit your model
linear_model.fit(poly_x, y)


def plotting_learning_curves(model, x, y):
    # TODO: split your data into train and validation (8:2)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    len_train = len(x_train)

    train_errors = []
    validation_errors = []

    for i in range(1, len_train):
        model.fit(x_train[:i], y_train[:i])
        pred_train = model.predict(x_train[:i])
        pred_val = model.predict(x_val)

        # TODO: get mean squared error of train data
        train_error = mse(y_train[:i], pred_train)

        # TODO: get mean squared error of validation data
        validation_error = mse(y_val, pred_val)

        train_errors.append(train_error)
        validation_errors.append(validation_error)

    # plotting part
    plt.plot(np.sqrt(train_errors), 'r', label="Train")
    plt.plot(np.sqrt(validation_errors), 'b', label="Validation")

    plt.xlabel("size of data")
    plt.ylabel("loss")

    plt.legend(
        loc='upper right',
        shadow=True,
        fancybox=True,
        borderpad=1  # border padding of the legend box
    )

    plt.savefig("./Image_Output/Polynomial_Regression.png")


plotting_learning_curves(linear_model, x, y)
