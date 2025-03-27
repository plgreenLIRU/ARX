import numpy as np
from scipy.sparse.linalg import cg

class Regressor:

    def __init__(self, N_AR=0):
        if not isinstance(N_AR, int):
            raise ValueError("N_AR must be an integer")
        self.N_AR = N_AR

    def prepare_arx_data(self, X, Y):

        # Ensure Y is a column vector
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        # Initialise auto-regressive features, exogenous features and targets
        ar_features, exog_features, targets = [], [], []

        # Create ARX data-matrix
        for t in range(self.N_AR, len(Y)):
            ar_features.append(Y[t-self.N_AR:t, 0])     # AR terms
            exog_features.append(X[t])       # Exogenous inputs
            targets.append(Y[t])             # Target value
        X_hat = np.hstack([exog_features, ar_features])
        Y_hat = np.array(targets).reshape(-1, 1)  # Ensure Y_hat is a 2D column vector

        return X_hat, Y_hat

    def train(self, X, Y):

        # Ensure Y is a column vector
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        # Size checks
        self.N, self.D = np.shape(X)
        assert Y.shape == (self.N, 1)
        if self.N_AR > 0:
            X, Y = self.prepare_arx_data(X, Y)

        # Form matrices
        A = X.T @ X
        b = X.T @ Y

        # Solve for theta using conjugate gradient
        theta, info = cg(A, b)
        self.theta = np.vstack(theta)

    def predict(self, X, y0=None):

        assert np.shape(X)[1] == self.D

        if self.N_AR == 0:
            Y = X @ self.theta
        else:
            assert len(y0) == self.N_AR

            Y = []
            for t in range(self.N_AR, np.shape(X)[0] + self.N_AR):

                if t == self.N_AR:
                    x = np.hstack([X[0], y0])
                else:
                    x[:self.D] = X[t - self.N_AR]
                    x[self.D:] = np.roll(x[self.D:], 1)
                    x[-1] = y

                y = x @ self.theta                
                Y.append(y[0])

        return Y
