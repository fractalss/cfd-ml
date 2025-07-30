import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator


class PCARBF_ROM_Generic:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = None
        self.rbf = None

    def fit(self, X, parameters):
        self.pca = PCA(n_components=self.n_components or min(X.shape))
        X_pca = self.pca.fit_transform(X)
        self.rbf = RBFInterpolator(parameters, X_pca, kernel='gaussian', epsilon=0.5, neighbors=1, smoothing=1e-6)
        return X_pca

    def predict(self, new_params):
        X_pca_new = self.rbf(new_params)
        return self.pca.inverse_transform(X_pca_new)