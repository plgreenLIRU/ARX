import numpy as np
from scipy.sparse.linalg import cg
from sklearn.linear_model import LinearRegression as SK_LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.cluster import KMeans

class Base:

    def __init__(self, N_AR, basis=None):
            """
            Initialises the Regressor object.

            Parameters:
            N_AR (int): Number of auto-regressive terms to include. Default is 0.
            """
            if not isinstance(N_AR, int):
                raise ValueError("N_AR must be an integer")
            self.N_AR = N_AR
            self.basis = basis

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
        # Check shapes
        assert y.shape == (X.shape[0],)

        # Initialise auto-regressive features, exogenous features and targets
        ar_features, exog_features, targets = [], [], []

        # Create ARX data-matrix
        for t in range(self.N_AR, len(y)):
            ar_features.append(y[t-self.N_AR:t])     # AR terms
            exog_features.append(X[t])       # Exogenous inputs
            targets.append(y[t])             # Target value
        X_hat = np.hstack([exog_features, ar_features])
        y_hat = np.array(targets)

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

        # Apply basis function
        if self.basis is None:
            Phi = X
        if self.basis is "se":
            Phi = self._se_basis(X, self.centres)        

        if self.N_AR == 0:
            y_pred = self.model.predict(Phi)
        else:
            assert len(y0) == self.N_AR

            y_pred = []
            for t in range(self.N_AR, np.shape(Phi)[0] + self.N_AR):

                # First time step
                if t == self.N_AR:
                    u = np.hstack([Phi[0], y0])
                    
                # Remaining time steps
                else:
                    u[:self.D] = Phi[t - self.N_AR]
                    u[self.D:] = np.roll(u[self.D:], 1)
                    u[-1] = y

                y = self.model.predict(u.reshape(1, -1))[0]               
                y_pred.append(y)

            # Finish by converting Y to array
            y_pred = np.array(y_pred)

        return y_pred

    def _se_basis(self, X, centres, width=1):
        dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        return np.exp(-0.5 * (dists / width) ** 2)

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
        # Size checks
        self.N, self.D = np.shape(X)
        assert y.shape == (self.N,)
        if self.N_AR > 0:
            X, y = self._prepare_arx_data(X, y)

        # Initialise
        self.model = SK_LinearRegression(positive=positive)
        
        # Apply basis function
        if self.basis is None:
            Phi = X
        if self.basis is "se":
            n_clusters = 10
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            kmeans.fit(X)
            self.centres = kmeans.cluster_centers_
            Phi = self._se_basis(X, self.centres)
        
        # Train model parameters
        self.model.fit(Phi, y)

class LinearBayes(Base):

    def train(self, X, y):

        # Size checks
        self.N, self.D = np.shape(X)
        assert y.shape == (self.N,)
        if self.N_AR > 0:
            X, y = self._prepare_arx_data(X, y)

        self.model = BayesianRidge()
        self.model.fit(X, y)

    def predict(self, X, y0=None, N_MC=100):

        assert np.shape(X)[1] == self.D

        if self.N_AR == 0:
            y_mean, y_std = self.model.predict(X, return_std=True)
            return y_mean, y_std
        else:
            assert len(y0) == self.N_AR
            Y_samples = np.zeros([np.shape(X)[0], N_MC])
            for t in range(self.N_AR, np.shape(X)[0] + self.N_AR):

                # First time step
                if t == self.N_AR:
                    U = np.tile(np.hstack([X[0], y0]), (N_MC, 1))
                    
                # Remaining time steps
                else:
                    U[:, :self.D] = np.tile(X[t - self.N_AR], (N_MC, 1))
                    U[:, self.D:] = np.roll(U[:, self.D:], shift=1, axis=1)
                    U[:, -1] = Y_samples[t - self.N_AR - 1]

                Y_mean, Y_std = self.model.predict(U, return_std=True)
                Y_samples[t - self.N_AR] = np.random.normal(loc=Y_mean, scale=Y_std)

            return Y_samples
