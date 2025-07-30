import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class PCARBF_ROM_Generic:
    def __init__(self, n_components=None, use_deltas=False, kernel_primary='gaussian', kernel_others='gaussian'):
        self.n_components = n_components
        self.use_deltas = use_deltas
        self.kernel_primary = kernel_primary
        self.kernel_others = kernel_others
        self.pca = None
        self.rbfs = []
        self.baseline_field = None
        self.X_pca_train = None
        self.parameters_train = None
        self.scaler = None
        self.param_scaler = None

    def fit(self, X, parameters):
        if self.use_deltas:
            self.baseline_field = X.mean(axis=0)
            X_residual = X - self.baseline_field
        else:
            self.baseline_field = np.zeros_like(X[0])
            X_residual = X

        if self.n_components is None:
            full_pca = PCA()
            full_pca.fit(X_residual)
            cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
            self.n_components = np.searchsorted(cumulative_variance, 0.999) + 1
            print(f"[INFO] Chosen number of components to retain =99.9% variance: {self.n_components}")

        self.pca = PCA(n_components=self.n_components)
        X_pca = self.pca.fit_transform(X_residual)
        self.X_pca_train = X_pca
        self.parameters_train = parameters

        print(f"[DEBUG] PCA fit completed. Shape of X_pca: {X_pca.shape}")
        print(f"[DEBUG] First few PCA coefficients:\n{X_pca[:5]}")

        self.scaler = StandardScaler().fit(X_pca)
        X_pca_scaled = self.scaler.transform(X_pca)

        self.param_scaler = StandardScaler().fit(parameters)
        parameters_scaled = self.param_scaler.transform(parameters)

        self.rbfs = []
        for i in range(self.n_components):
            kernel = self.kernel_primary if i == 0 else self.kernel_others
            epsilon = 0.5 if i == 0 else 2.0
            smoothing = 1e-2 if i == 0 else 1e-1
            neighbors = 1 if i == 0 else 20
            rbf = RBFInterpolator(parameters_scaled, X_pca_scaled[:, i], kernel=kernel, neighbors=neighbors, epsilon=epsilon, smoothing=smoothing)
            self.rbfs.append(rbf)

        return X_pca

    def predict(self, new_params):
        new_params_scaled = self.param_scaler.transform(new_params)
        X_pca_scaled = np.column_stack([rbf(new_params_scaled) for rbf in self.rbfs])
        X_pca_inverse = self.scaler.inverse_transform(X_pca_scaled)
        reconstructed_residual = self.pca.inverse_transform(X_pca_inverse)
        reconstructed = reconstructed_residual + self.baseline_field if self.use_deltas else reconstructed_residual

        return reconstructed

    def interpolate(self, new_params):
        """
        Returns interpolated (but still scaled) PCA coefficients at new parameter values.
        """
        new_params_scaled = self.param_scaler.transform(new_params)
        X_pca_scaled = np.column_stack([rbf(new_params_scaled) for rbf in self.rbfs])
        return X_pca_scaled

    def test_reconstruction_from_training(self, index=0):
        X_pca_single = self.X_pca_train[index:index+1]
        reconstructed = self.pca.inverse_transform(X_pca_single)
        return reconstructed + self.baseline_field if self.use_deltas else reconstructed

    def compare_pca_coefficients(self, test_param, true_index=0):
        test_param_scaled = self.param_scaler.transform(test_param)
        interpolated = np.column_stack([rbf(test_param_scaled) for rbf in self.rbfs])
        true_scaled = self.scaler.transform(self.X_pca_train[true_index:true_index+1])
        mae = np.mean(np.abs(interpolated - true_scaled), axis=0)

        plt.figure(figsize=(10, 4))
        plt.plot(true_scaled.flatten(), 'o-', label='True Coefficients (Scaled)')
        plt.plot(interpolated.flatten(), 'x--', label='Interpolated Coefficients (Scaled)')
        plt.title("Comparison of Scaled PCA Coefficients")
        plt.xlabel("PCA Mode Index")
        plt.ylabel("Coefficient Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print(f"[INFO] Mean Absolute Error per PCA mode: {mae}")
