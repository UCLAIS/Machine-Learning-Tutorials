"""
<Ridge, Lasso, ElasticNet with scikit-learn>

1. load Ridge model with
    sklearn.linear_model.Ridge(alpha)
        alpha: scalar value

2.  load Lasso with
    sklearn.linear_model.Lasso(alpha)
        alpha: scalar value

3. load ElasticNet with
    sklearn.linear_model.ElasticNet(alpha, l1_ratio)
        alpha: scalar value
        l1_ratio: ratio for L1 norm

[Model].fit(x, y)
[Model].predict(x)

reference: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
"""
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")

# Random generation of x and y
x = 5 * np.random.rand(100, 1)
y = 3 * x + 5 * np.random.rand(100, 1)

# TODO: Load Ridge and Train it!
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(x, y)

# TODO: Load Lasso and Train it!
lasso_reg = Lasso(alpha=0.05)
lasso_reg.fit(x, y)

# TODO: Load ElasticNet and Train it!
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_net.fit(x, y)

# TODO: Predict!
ridge_y_pred = ridge_reg.predict(x)
lasso_y_pred = lasso_reg.predict(x)
elastic_y_pred = elastic_net.predict(x)

# Graph it!
plt.title("Ridge vs. Lasso vs. ElasticNet")
plt.scatter(x, y)

plt.plot(x, ridge_y_pred, color='green', label='Ridge')
plt.plot(x, lasso_y_pred, color='red', label='Lasso')
plt.plot(x, elastic_y_pred, color='blue', label='ElasticNet')

plt.legend(
        loc='upper left',
        shadow=True,
        fancybox=True,
        borderpad=1    # border padding of the legend box
    )

plt.savefig("./Image_Output/Ridge_Lasso_ElasticNet.png")
