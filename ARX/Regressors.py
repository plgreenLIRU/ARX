import numpy as np
from scipy.sparse.linalg import cg
from sklearn.linear_model import LinearRegression as SK_LinearRegression
from sklearn.neural_network import MLPRegressor as SK_MLPRegressor

class Base:

    def __init__(self, N_AR=0):
            """
            Initialises the Regressor object.

            Parameters:
            N_AR (int): Number of auto-regressive terms to include. Default is 0.
            """
            if not isinstance(N_AR, int):
                raise ValueError("N_AR must be an integer")
            self.N_AR = N_AR

    def _prepare_arx_data(self, X, y):
        """
        Prepares the ARX (Auto-Regressive with eXogenous inputs) data matrix.

        Parameters:
        X (numpy.ndarray): Exogenous input data of shape (N, D).
        y (numpy.ndarray): Target data of shape (N,).

        Returns:
        tuple: A tuple (X_hat, Y_hat) where:
            - X_hat (numpy.ndarray): Combined AR and exogenous features.
            - Y_hat (numpy.ndarray): Target values as a column vector.
        """
        # Ensure y is a column vector
        y = y.reshape(-1, 1) if y.ndim == 1 else y

        # Initialise auto-regressive features, exogenous features and targets
        ar_features, exog_features, targets = [], [], []

        # Create ARX data-matrix
        for t in range(self.N_AR, len(y)):
            ar_features.append(y[t-self.N_AR:t, 0])     # AR terms
            exog_features.append(X[t])       # Exogenous inputs
            targets.append(y[t])             # Target value
        X_hat = np.hstack([exog_features, ar_features])
        y_hat = np.array(targets).reshape(-1, 1)  # Ensure Y_hat is a 2D column vector

        return X_hat, y_hat

    def predict(self, X, y0=None):
        """
        Predicts target values using the trained model.

        Parameters:
        X (numpy.ndarray): Input data of shape (N, D).
        y0 (numpy.ndarray, optional): Initial auto-regressive terms of shape (N_AR,).
                                      Required if N_AR > 0.

        Returns:
        numpy.ndarray: Predicted target values.
        """
        assert np.shape(X)[1] == self.D
        

        if self.N_AR == 0:
            y_pred = self.model.predict(X)
        else:
            assert len(y0) == self.N_AR

            y_pred = []
            for t in range(self.N_AR, np.shape(X)[0] + self.N_AR):

                # First time step
                if t == self.N_AR:
                    x = np.hstack([X[0], y0])
                    
                # Remaining time steps
                else:
                    x[:self.D] = X[t - self.N_AR]
                    x[self.D:] = np.roll(x[self.D:], 1)
                    x[-1] = y

                y = self.model.predict(x.reshape(1, -1))[0]               
                y_pred.append(y)

            # Finish by converting Y to array
            y_pred = np.array(y_pred)

        return np.vstack(y_pred)

class Linear(Base):

    def train(self, X, y, positive=False):
        """
        Trains the regressor using the provided data.

        Parameters:
        X (numpy.ndarray): Input data of shape (N, D).
        y (numpy.ndarray): Target data of shape (N,).

        Returns:
        None
        """
        # Ensure y is a column vector
        y = y.reshape(-1, 1) if y.ndim == 1 else y

        # Size checks
        self.N, self.D = np.shape(X)
        assert y.shape == (self.N, 1)
        if self.N_AR > 0:
            X, y = self._prepare_arx_data(X, y)

        self.model = SK_LinearRegression(positive=positive)
        self.model.fit(X, y)
