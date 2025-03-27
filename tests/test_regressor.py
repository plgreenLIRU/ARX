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

    # Check if the estimated theta is close to the true theta
    assert np.allclose(regressor.theta, true_theta)