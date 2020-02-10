"""
Let's do Linear Regression using scikit-learn library

1. Load LR model with
    linear_model = sklearn.linear_model.LinearRegression()

2. Train your model with
    LinearRegression.fit(x, y)

3. Predict y_hat with
    LinearRegression.predict(x)

reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")


# Random creation of x and y
x = 5 * np.random.rand(100, 1)
y = 3 * x + 5 * np.random.rand(100, 1)

# TODO: Load LR model!
lr_model = LinearRegression()
# TODO: Train!
lr_model.fit(x, y)

# TODO: Predict!
predicted = lr_model.predict(x)

# Visualise
fig, ax = plt.subplots(1, 2, figsize=(16, 7))

ax[0].scatter(x, y)
ax[1].scatter(x, y)
ax[1].plot(x, predicted, color='r')

ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')

fig.savefig("./Image_Output/Linear_Regression.png")
