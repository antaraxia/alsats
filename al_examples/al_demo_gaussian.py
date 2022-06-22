import numpy as np
from modAL.models import ActiveLearner
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
import matplotlib.pyplot as plt


def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]


def al_demo():
    X = np.random.choice(np.linspace(0, 20, 10000), size=200,
                         replace=False).flatten()
    X = np.sort(X).reshape((-1,1))
  
    print(X)
    y = np.sin(X) + np.random.normal(scale=0.3, size=X.shape)

    n_initial = 5
    initial_idx = np.random.choice(range(len(X)),
                                   size=n_initial,
                                   replace=False)
    X_training, y_training = X[initial_idx], y[initial_idx]

    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
             + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

    regressor = ActiveLearner(
        estimator=GaussianProcessRegressor(kernel=kernel),
        query_strategy=GP_regression_std,
        X_training=X_training.reshape(-1, 1),
        y_training=y_training.reshape(-1, 1))

    n_queries = 20
    for idx in range(n_queries):
        query_idx, query_instance = regressor.query(X)
        regressor.teach(X[query_idx].reshape(1, -1),
                        y[query_idx].reshape(1, -1))

    y_pred = regressor.estimator.predict(X)
    plt.scatter(X, y,label='y')
    plt.plot(X, y_pred,label='y_pred')
    plt.title('Predicted versus Learned value in {} iterations'.format(n_queries))
    plt.legend()
    plt.show()
