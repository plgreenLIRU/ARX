import numpy as np
from scipy.sparse.linalg import cg

class Regressor:

    def __init__(self, N_AR):
        if not isinstance(N_AR, int):
            raise ValueError("N_AR must be an integer")
        self.N_AR = N_AR

    def prepare_arx_data(self, X, Y):

        # Initialise auto-regressive features, exogenous features and targets
        ar_features, exog_features, targets = [], [], []

        # Create ARX data-matrix
        for t in range(self.N_AR, len(Y)):
            ar_features.append(Y[t-self.N_AR:t])     # AR terms
            exog_features.append(X[t])       # Exogenous inputs
            targets.append(Y[t])             # Target value
        X_hat = np.hstack([ar_features, exog_features])
        Y_hat = np.array(targets).reshape(-1, 1)  # Ensure Y_hat is a 2D column vector

        return X_hat, Y_hat

    def train(self, X, Y):

        # Ensure Y is a column vector
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        # Size checks
        self.N, self.D = np.shape(X)
        assert Y.shape == (self.N, 1), "Y must be an N x 1 column vector"

        if self.N_AR > 0:
            pass

        # Form matrices
        A = X.T @ X
        b = X.T @ Y

        # Solve for theta using conjugate gradient
        theta, info = cg(A, b)
        self.theta = np.vstack(theta)

    def predict(self, X):
        return X @ self.theta
