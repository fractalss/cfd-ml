import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt

class PCARBF_ROM_Generic:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = None
        self.X_pca_train = None
        self.parameters_train = None
        self.scaler = None
        self.param_scaler = None
        self.rbfs = []

    def fit(self, X, parameters):
        if self.n_components is None:
            full_pca = PCA()
            full_pca.fit(X)
            cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
            self.n_components = np.searchsorted(cumulative_variance, 0.999) + 1
            print(f"[INFO] Chosen number of components to retain =99.9% variance: {self.n_components}")

        self.pca = PCA(n_components=self.n_components)
        X_pca = self.pca.fit_transform(X)
        self.X_pca_train = X_pca
        self.parameters_train = parameters

        print(f"[DEBUG] PCA fit completed. Shape of X_pca: {X_pca.shape}")
        print(f"[DEBUG] First few PCA coefficients:\n{X_pca[:5]}")

        self.scaler = StandardScaler().fit(X_pca)
        X_pca_scaled = self.scaler.transform(X_pca)
        self.X_pca_scaled = X_pca_scaled

        self.param_scaler = StandardScaler().fit(parameters)
        self.parameters_scaled = self.param_scaler.transform(parameters)

        self.rbfs = []
        for i in range(self.n_components):
            rbf = RBFInterpolator(
                self.parameters_scaled,
                X_pca_scaled[:, i],
                kernel='gaussian',
                epsilon=1.0,
                smoothing=1e-3
            )
            self.rbfs.append(rbf)

        return X_pca

    def predict(self, param):
        param_array = np.array(param).reshape(1, -1)
        param_scaled = self.param_scaler.transform(param_array)

        X_pca_interp_scaled = np.column_stack([
            rbf(param_scaled) for rbf in self.rbfs
        ])

        X_pca_interp = self.scaler.inverse_transform(X_pca_interp_scaled)
        X_reconstructed = self.pca.inverse_transform(X_pca_interp)

        return X_reconstructed

    def compare_pca_coefficients(self, true_index=0):
        true_scaled = self.scaler.transform(self.X_pca_train[true_index:true_index+1])
        interp_scaled = np.column_stack([
            rbf(self.parameters_scaled[true_index:true_index+1]) for rbf in self.rbfs
        ])
        mae = np.mean(np.abs(true_scaled - interp_scaled), axis=0)

        plt.figure(figsize=(10, 4))
        plt.plot(true_scaled.flatten(), 'o-', label='True Coefficients (Scaled)')
        plt.plot(interp_scaled.flatten(), 'x--', label='Interpolated Coefficients (Scaled)')
        plt.title("Comparison of Scaled PCA Coefficients")
        plt.xlabel("PCA Mode Index")
        plt.ylabel("Coefficient Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print(f"[INFO] Mean Absolute Error per PCA mode: {mae}")
