import numpy as np
from ARXLR.Regressor import Regressor

def test_linear_regression():

    # Create a simple 3D linear regression problem
    np.random.seed(42)  # For reproducibility
    D = 3  # Number of features
    N = 100  # Number of samples

    # Generate random features
    X = np.random.rand(N, D)

    # True coefficients
    true_theta = np.vstack(np.array([2.0, -1.0, 0.5]))

    # Generate target values with some noise
    Y = X @ true_theta

    # Initialize and train the regressor
    regressor = Regressor()
    regressor.train(X, Y)
    Y_pred = regressor.predict(X)

    # Check if the estimated theta is close to the true theta
    assert np.allclose(regressor.theta, true_theta)

    # Check predictions
    assert np.allclose(Y, Y_pred)

def test_prepare_arx_data():

    np.random.seed(42)  # For reproducibility
    D = 3  # Number of features
    N = 4  # Number of samples

    # Generate random data
    X = np.random.rand(N, D)
    Y = np.random.rand(N)

    regressor = Regressor(N_AR=2)
    X_hat, Y_hat = regressor.prepare_arx_data(X, Y)

    # Check shapes
    assert np.shape(X_hat) == (2, 5)
    assert np.shape(Y_hat) == (2, 1)

    # Check values
    assert np.allclose(X_hat[0, :], np.hstack([X[2, :], Y[0:2]]))
    assert np.allclose(X_hat[1, :], np.hstack([X[3, :], Y[1:3]]))
