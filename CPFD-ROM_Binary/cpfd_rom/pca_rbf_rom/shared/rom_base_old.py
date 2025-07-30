import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator

class PCARBF_ROM_Generic:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = None
        self.rbf = None

    def fit(self, X, parameters):
        # Step 1: If n_components is None, determine number of components to retain 90% variance
        if self.n_components is None:
            full_pca = PCA()
            full_pca.fit(X)
            cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
            self.n_components = np.searchsorted(cumulative_variance, 0.99) + 1
            print(f"[INFO] Chosen number of components to retain =90% variance: {self.n_components}")

        # Step 2: Fit PCA with chosen number of components
        self.pca = PCA(n_components=self.n_components)
        X_pca = self.pca.fit_transform(X)
        print(f"[DEBUG] PCA fit completed. Shape of X_pca: {X_pca.shape}")
        print(f"[DEBUG] First few PCA coefficients:\n{X_pca[:5]}")

        # Step 3: Fit RBF interpolator on PCA-transformed data
        # self.rbf = RBFInterpolator(parameters, X_pca, kernel='gaussian', epsilon=2.0, neighbors=10, smoothing=1e-1)
        self.rbf = RBFInterpolator(parameters, X_pca, kernel='thin_plate_spline', smoothing=1e-1)
        return X_pca

    def predict(self, new_params):
        X_pca_new = self.rbf(new_params)
        print(f"[DEBUG] New parameter: {new_params}")
        print(f"[DEBUG] Interpolated PCA coefficients: {X_pca_new}")
        reconstructed = self.pca.inverse_transform(X_pca_new)
        print(f"[DEBUG] Reconstructed field shape: {reconstructed.shape}")
        return self.pca.inverse_transform(X_pca_new)
