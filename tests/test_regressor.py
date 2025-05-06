import numpy as np
from ARX.Regressors import Linear, ANN
from sklearn.neural_network import MLPRegressor

def test_linear_regression():
    """
    Tests the Regressor's ability to solve a simple linear regression problem.

    This test creates a 3D linear regression problem with known coefficients,
    trains the Regressor on the data, and verifies that the estimated coefficients
    and predictions are close to the true values.
    """
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
    regressor = Linear()
    regressor.train(X, Y)
    Y_pred = regressor.predict(X)

    # Check if the estimated theta is close to the true theta
    assert np.allclose(regressor.model.coef_, true_theta[:, 0])

    # Check predictions
    assert np.allclose(Y, Y_pred)

def test_prepare_arx_data():
    """
    Tests the `_prepare_arx_data` method for generating ARX data matrices.

    This test verifies that the method correctly combines exogenous inputs
    and auto-regressive terms into the ARX data matrix, and ensures the
    output shapes and values are as expected.
    """
    np.random.seed(42)  # For reproducibility
    D = 3  # Number of features
    N = 4  # Number of samples

    # Generate random data
    X = np.random.rand(N, D)
    Y = np.random.rand(N)

    regressor = Linear(N_AR=2)
    X_hat, Y_hat = regressor._prepare_arx_data(X, Y)

    # Check shapes
    assert np.shape(X_hat) == (2, 5)
    assert np.shape(Y_hat) == (2, 1)

    # Check values
    assert np.allclose(X_hat[0, :], np.hstack([X[2, :], Y[0:2]]))
    assert np.allclose(X_hat[1, :], np.hstack([X[3, :], Y[1:3]]))

def test_arx_linear_regression():
    """
    Tests the Regressor's ability to handle ARX models with auto-regressive terms.

    This test generates data with both exogenous inputs and auto-regressive
    components, trains the Regressor, and verifies that the estimated coefficients
    and predictions are close to the true values.
    """
    np.random.seed(42)  # For reproducibility
    D = 3  # Number of exogenous features
    N = 100  # Number of samples
    N_AR = 2  # Number of auto-regressive components

    # Generate random exogenous inputs
    X = np.random.rand(N, D)

    # True coefficients for exogenous inputs and AR components
    theta_exog = np.array([2.0, -1.0, 0.5]).reshape(-1, 1)  # Coefficients for exogenous inputs
    theta_ar = np.array([0.2, -0.1]).reshape(-1, 1)         # Coefficients for AR components
    true_theta = np.vstack([theta_exog, theta_ar])

    # Generate target values with AR components
    Y = np.zeros(N)
    for t in range(N_AR, N):
        Y[t] = (
            X[t] @ theta_exog +  # Contribution from exogenous inputs
            Y[t-N_AR : t] @ theta_ar  # Contribution from AR components
        )

    # Initialize and train the regressor
    regressor = Linear(N_AR=N_AR)
    regressor.train(X, Y)

    # Check if the estimated theta is close to the true theta
    assert np.allclose(regressor.model.coef_, true_theta[:, 0], atol=1e-2)

    # Check full model predictions
    Y_pred = regressor.predict(X[N_AR:], y0=Y[:N_AR])
    assert np.allclose(Y[N_AR:], Y_pred[:, 0])

def test_arx_ann():
    """
    ...
    """

    np.random.seed(42)  # For reproducibility
    D = 3  # Number of exogenous features
    N = 100  # Number of samples
    N_AR = 2  # Number of auto-regressive components

    # Generate random exogenous inputs
    X = np.random.rand(N, D)

    # Random initialization for Y beginning
    Y = np.zeros(N)
    Y[:N_AR] = np.random.randn(N_AR)

    # Fit it on dummy data to initialize weights (weights are randomly set based on random_state)
    ann = MLPRegressor(hidden_layer_sizes=(3), max_iter=1)    
    ann.fit(np.random.randn(3, D + N_AR), np.random.randn(3))

    # Generate autoregressive data using the untrained neural net
    for t in range(N_AR, N):
        x_ar = Y[t - N_AR:t]
        x_full = np.concatenate([X[t], x_ar])
        Y[t] = ann.predict(x_full.reshape(1, -1))[0]

    # Initialize and train the ann model
    regressor = ANN(N_AR=N_AR)
    regressor.train(X, Y, hidden_layer_sizes=(3))

    # Check full model predictions
    Y_pred = regressor.predict(X[N_AR:], y0=Y[:N_AR])
    assert np.allclose(Y[N_AR:], Y_pred[:, 0])
