import numpy as np
from scipy.sparse.linalg import cg

class Regressor:

    def __init__(self):
        pass

    def train(self, X, Y):
        # Ensure Y is a column vector
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        # Size checks
        self.N, self.D = np.shape(X)
        assert Y.shape == (self.N, 1), "Y must be an N x 1 column vector"

        # Form matrices
        A = X.T @ X
        b = X.T @ Y

        # Solve for theta using conjugate gradient
        theta, info = cg(A, b)
        self.theta = np.vstack(theta)
