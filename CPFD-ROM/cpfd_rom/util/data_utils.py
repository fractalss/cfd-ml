def crop_to_divisible_by_4(X, factor=4):
    """
    Crop the second dimension of X to be divisible by the given factor (default=4).
    """
    n_features = X.shape[1]
    if n_features % factor != 0:
        crop_len = n_features - (n_features // factor) * factor
        X = X[:, :-crop_len]
        print(f"[INFO] Cropped input from {n_features} to {X.shape[1]} to match model architecture.")
    return X
